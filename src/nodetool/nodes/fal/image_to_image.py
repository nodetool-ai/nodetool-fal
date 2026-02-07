from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.metadata.types import ImageRef, VideoRef
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


class FluxSchnellRedux(FALNode):
    """
    FLUX.1 [schnell] Redux enables rapid transformation of existing images with high-quality style transfers and modifications using the fast FLUX.1 schnell model.
    image, transformation, style-transfer, fast, flux, redux

    Use cases:
    - Transform images with artistic style transfers
    - Apply quick modifications to photos
    - Create image variations for rapid iteration
    - Generate stylized versions of existing images
    - Produce fast image transformations
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


    num_images: int = Field(
        default=1, description="The number of images to generate (1-4)"
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3, description="The size of the generated image"
    )
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="Acceleration speed: 'none', 'regular', or 'high'"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="Output format (jpeg or png)"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to generate an image from."
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
    num_inference_steps: int = Field(
        default=4, description="The number of inference steps to perform (1-50)"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "num_images": self.num_images,
            "image_size": self.image_size.value,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux/schnell/redux",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "image_size", "num_inference_steps"]

class FluxDevRedux(FALNode):
    """
    FLUX.1 [dev] Redux provides advanced image transformation capabilities with superior quality and more control over the style transfer process.
    image, transformation, style-transfer, development, flux, redux

    Use cases:
    - Transform images with advanced quality controls
    - Create customized image variations with guidance
    - Apply precise style modifications
    - Generate high-quality artistic transformations
    - Produce refined image edits with better prompt adherence
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


    num_images: int = Field(
        default=1, description="The number of images to generate (1-4)"
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3, description="The size of the generated image"
    )
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="The speed of the generation. The higher the speed, the faster the generation."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="Output format (jpeg or png)"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to generate an image from."
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
        default=3.5, description="How strictly to follow the image structure (1-20)"
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform (1-50)"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "num_images": self.num_images,
            "image_size": self.image_size.value,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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
            application="fal-ai/flux/dev/redux",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "image_size", "guidance_scale"]

class FluxProRedux(FALNode):
    """
    FLUX.1 Pro Redux delivers professional-grade image transformations with the highest quality and safety controls for commercial use.
    image, transformation, style-transfer, professional, flux, redux

    Use cases:
    - Create professional-quality image transformations
    - Apply commercial-grade style transfers
    - Generate high-fidelity image variations
    - Produce brand-safe image modifications
    - Transform images for production use
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
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate (1-4)"
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3, description="The size of the generated image"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="Output format (jpeg or png)"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The image URL to generate an image from. Needs to match the dimensions of the mask."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    safety_tolerance: SafetyTolerance = Field(
        default=SafetyTolerance.VALUE_2, description="Safety tolerance level (1-6, higher is stricter)"
    )
    guidance_scale: float = Field(
        default=3.5, description="How strictly to follow the image structure (1-20)"
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform (1-50)"
    )
    seed: int = Field(
        default=-1, description="Seed for reproducible results. Use -1 for random"
    )
    enhance_prompt: bool = Field(
        default=False, description="Whether to enhance the prompt for better results."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size.value,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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
            application="fal-ai/flux-pro/v1/redux",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "image_size", "guidance_scale"]

class IdeogramV2Edit(FALNode):
    """
    Transform existing images with Ideogram V2's editing capabilities. Modify, adjust, and refine images while maintaining high fidelity with precise prompt and mask control.
    image, editing, inpainting, mask, ideogram, transformation

    Use cases:
    - Edit specific parts of images with precision
    - Create targeted image modifications using masks
    - Refine and enhance image details
    - Generate contextual image edits
    - Replace or modify masked regions
    """

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
        default="", description="The prompt to fill the masked part of the image"
    )
    style: Style = Field(
        default=Style.AUTO, description="Style of generated image (auto, general, realistic, design, render_3D, anime)"
    )
    expand_prompt: bool = Field(
        default=True, description="Whether to expand the prompt with MagicPrompt functionality"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The image URL to generate an image from. Needs to match the dimensions of the mask."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    seed: int = Field(
        default=-1, description="Seed for reproducible results. Use -1 for random"
    )
    mask_url: ImageRef = Field(
        default=ImageRef(), description="The mask URL to inpaint the image. Needs to match the dimensions of the input image."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        mask_url_base64 = await context.image_to_base64(self.mask_url)
        arguments = {
            "prompt": self.prompt,
            "style": self.style.value,
            "expand_prompt": self.expand_prompt,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "seed": self.seed,
            "mask_url": f"data:image/png;base64,{mask_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ideogram/v2/edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image", "mask"]

class IdeogramV2Remix(FALNode):
    """
    Reimagine existing images with Ideogram V2's remix feature. Create variations and adaptations while preserving core elements through prompt guidance and strength control.
    image, remix, variation, creativity, ideogram, adaptation

    Use cases:
    - Create artistic variations of images
    - Generate style-transferred versions
    - Produce creative image adaptations
    - Transform images while preserving key elements
    - Generate alternative interpretations
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
        default="", description="The prompt to remix the image with"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1, description="The aspect ratio of the generated image"
    )
    style: Style = Field(
        default=Style.AUTO, description="Style of generated image (auto, general, realistic, design, render_3D, anime)"
    )
    expand_prompt: bool = Field(
        default=True, description="Whether to expand the prompt with MagicPrompt functionality"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The image URL to remix"
    )
    strength: float = Field(
        default=0.8, description="Strength of the input image in the remix (0-1, higher = more variation)"
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    seed: int = Field(
        default=-1, description="Seed for reproducible results. Use -1 for random"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "style": self.style.value,
            "expand_prompt": self.expand_prompt,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "strength": self.strength,
            "sync_mode": self.sync_mode,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ideogram/v2/remix",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image", "strength"]

class IdeogramV3Edit(FALNode):
    """
    Transform images with Ideogram V3's enhanced editing capabilities. Latest generation editing with improved quality, control, and style consistency.
    image, editing, inpainting, mask, ideogram, v3

    Use cases:
    - Edit images with the latest Ideogram technology
    - Apply high-fidelity masked edits
    - Generate professional image modifications
    - Create precise content-aware fills
    - Refine image details with advanced controls
    """

    class RenderingSpeed(Enum):
        """
        The rendering speed to use.
        """
        TURBO = "TURBO"
        BALANCED = "BALANCED"
        QUALITY = "QUALITY"


    prompt: str = Field(
        default="", description="The prompt to fill the masked part of the image"
    )
    num_images: int = Field(
        default=1, description="Number of images to generate."
    )
    style_preset: str = Field(
        default="", description="Style preset for generation. The chosen style preset will guide the generation."
    )
    expand_prompt: bool = Field(
        default=True, description="Determine if MagicPrompt should be used in generating the request or not."
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
    image_url: ImageRef = Field(
        default=ImageRef(), description="The image URL to generate an image from. MUST have the exact same dimensions (width and height) as the mask image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    seed: int = Field(
        default=-1, description="Seed for reproducible results. Use -1 for random"
    )
    image_urls: ImageRef = Field(
        default=ImageRef(), description="A set of images to use as style references (maximum total size 10MB across all style references). The images should be in JPEG, PNG or WebP format"
    )
    mask_url: ImageRef = Field(
        default=ImageRef(), description="The mask URL to inpaint the image. MUST have the exact same dimensions (width and height) as the input image."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        image_urls_base64 = await context.image_to_base64(self.image_urls)
        mask_url_base64 = await context.image_to_base64(self.mask_url)
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "style_preset": self.style_preset,
            "expand_prompt": self.expand_prompt,
            "rendering_speed": self.rendering_speed.value,
            "style_codes": self.style_codes,
            "color_palette": self.color_palette,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "seed": self.seed,
            "image_urls": f"data:image/png;base64,{image_urls_base64}",
            "mask_url": f"data:image/png;base64,{mask_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ideogram/v3/edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image", "mask"]

class FluxProFill(FALNode):
    """
    FLUX.1 Pro Fill provides professional inpainting and outpainting capabilities. Generate or modify image content within masked regions with precise prompt control.
    image, inpainting, outpainting, fill, flux, professional

    Use cases:
    - Fill masked regions with new content
    - Extend images beyond their boundaries (outpainting)
    - Remove unwanted objects and fill gaps
    - Generate content-aware image expansions
    - Create seamless image modifications
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
        default="", description="The prompt describing what to generate in the masked area"
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The image URL to generate an image from. Needs to match the dimensions of the mask."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    safety_tolerance: SafetyTolerance = Field(
        default=SafetyTolerance.VALUE_2, description="Safety tolerance level (1-6, higher is stricter)"
    )
    seed: int = Field(
        default=-1, description="Seed for reproducible results. Use -1 for random"
    )
    mask_url: ImageRef = Field(
        default=ImageRef(), description="The mask URL to inpaint the image. Needs to match the dimensions of the input image."
    )
    enhance_prompt: bool = Field(
        default=False, description="Whether to enhance the prompt for better results."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        mask_url_base64 = await context.image_to_base64(self.mask_url)
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "safety_tolerance": self.safety_tolerance.value,
            "seed": self.seed,
            "mask_url": f"data:image/png;base64,{mask_url_base64}",
            "enhance_prompt": self.enhance_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-pro/v1/fill",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image", "mask"]

class FluxProCanny(FALNode):
    """
    FLUX.1 Pro with Canny edge detection control. Generate images guided by edge maps for precise structural control while maintaining FLUX's quality.
    image, controlnet, canny, edges, flux, professional

    Use cases:
    - Generate images following edge structures
    - Transform images while preserving edges
    - Create controlled variations with edge guidance
    - Apply style transfers with structural constraints
    - Generate content from edge maps
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
        default="", description="The text prompt describing the desired output"
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3, description="The size of the generated image"
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
        default=3.5, description="How strictly to follow the prompt"
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform"
    )
    control_image_url: ImageRef = Field(
        default=ImageRef(), description="The control image URL to generate the Canny edge map from."
    )
    seed: int = Field(
        default=-1, description="Seed for reproducible results. Use -1 for random"
    )
    enhance_prompt: bool = Field(
        default=False, description="Whether to enhance the prompt for better results."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        control_image_url_base64 = await context.image_to_base64(self.control_image_url)
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "safety_tolerance": self.safety_tolerance.value,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "control_image_url": f"data:image/png;base64,{control_image_url_base64}",
            "seed": self.seed,
            "enhance_prompt": self.enhance_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-pro/v1/canny",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image", "control_strength"]

class FluxProDepth(FALNode):
    """
    FLUX.1 Pro with depth map control. Generate images guided by depth information for precise 3D structure control while maintaining FLUX's quality.
    image, controlnet, depth, 3d, flux, professional

    Use cases:
    - Generate images following depth structures
    - Transform images while preserving 3D composition
    - Create controlled variations with depth guidance
    - Apply style transfers with spatial constraints
    - Generate content from depth maps
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
        default="", description="The text prompt describing the desired output"
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3, description="The size of the generated image"
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
        default=3.5, description="How strictly to follow the prompt"
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform"
    )
    control_image_url: ImageRef = Field(
        default=ImageRef(), description="The control image URL to generate the depth map from."
    )
    seed: int = Field(
        default=-1, description="Seed for reproducible results. Use -1 for random"
    )
    enhance_prompt: bool = Field(
        default=False, description="Whether to enhance the prompt for better results."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        control_image_url_base64 = await context.image_to_base64(self.control_image_url)
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "safety_tolerance": self.safety_tolerance.value,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "control_image_url": f"data:image/png;base64,{control_image_url_base64}",
            "seed": self.seed,
            "enhance_prompt": self.enhance_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-pro/v1/depth",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image", "control_strength"]

class BriaEraser(FALNode):
    """
    Bria Eraser removes unwanted objects from images using intelligent inpainting. Seamlessly fill removed areas with contextually appropriate content.
    image, eraser, removal, inpainting, bria, cleanup

    Use cases:
    - Remove unwanted objects from photos
    - Clean up image backgrounds
    - Erase text or watermarks
    - Delete distracting elements
    - Create clean product shots
    """

    class MaskType(Enum):
        """
        You can use this parameter to specify the type of the input mask from the list. 'manual' opttion should be used in cases in which the mask had been generated by a user (e.g. with a brush tool), and 'automatic' mask type should be used when mask had been generated by an algorithm like 'SAM'.
        """
        MANUAL = "manual"
        AUTOMATIC = "automatic"


    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    preserve_alpha: bool = Field(
        default=False, description="If set to true, attempts to preserve the alpha channel of the input image."
    )
    mask_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the binary mask image that represents the area that will be cleaned."
    )
    mask_type: MaskType = Field(
        default=MaskType.MANUAL, description="You can use this parameter to specify the type of the input mask from the list. 'manual' opttion should be used in cases in which the mask had been generated by a user (e.g. with a brush tool), and 'automatic' mask type should be used when mask had been generated by an algorithm like 'SAM'."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="Input Image to erase from"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        mask_url_base64 = await context.image_to_base64(self.mask_url)
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "sync_mode": self.sync_mode,
            "preserve_alpha": self.preserve_alpha,
            "mask_url": f"data:image/png;base64,{mask_url_base64}",
            "mask_type": self.mask_type.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/eraser",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "mask"]

class BriaBackgroundReplace(FALNode):
    """
    Bria Background Replace swaps image backgrounds with new content. Intelligently separates subjects and generates contextually appropriate backgrounds.
    image, background, replacement, segmentation, bria

    Use cases:
    - Replace photo backgrounds with custom scenes
    - Create product shots with various backgrounds
    - Change image context while preserving subject
    - Generate professional portraits with studio backgrounds
    - Create marketing materials with branded backgrounds
    """

    prompt: str = Field(
        default="", description="Description of the new background to generate"
    )
    steps_num: int = Field(
        default=30, description="Number of inference steps."
    )
    sync_mode: bool = Field(
        default=False, description="If true, returns the image directly in the response (increases latency)."
    )
    seed: int = Field(
        default=4925634, description="Random seed for reproducibility."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt for background replacement."
    )
    image_url: ImageRef = Field(
        default="https://v3b.fal.media/files/b/0a8bea8c/Mztgx0NG3HPdby-4iPqwH_a_coffee_machine_standing_in_the_kitchen.png", description="Reference image (file or URL)."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "steps_num": self.steps_num,
            "sync_mode": self.sync_mode,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/replace-background",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ClarityUpscaler(FALNode):
    """
    Clarity Upscaler increases image resolution using AI-powered super-resolution. Enhance image quality, sharpness, and detail up to 4x scale.
    image, upscaling, enhancement, super-resolution, clarity

    Use cases:
    - Increase image resolution for printing
    - Improve clarity of low-quality images
    - Enhance textures and fine details
    - Prepare images for large displays
    - Restore detail in compressed images
    """

    prompt: str = Field(
        default="masterpiece, best quality, highres", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    resemblance: float = Field(
        default=0.6, description="The resemblance of the upscaled image to the original image. The higher the resemblance, the more the model will try to keep the original image. Refers to the strength of the ControlNet."
    )
    creativity: float = Field(
        default=0.35, description="The creativity of the model. The higher the creativity, the more the model will deviate from the prompt. Refers to the denoise strength of the sampling."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to upscale."
    )
    upscale_factor: float = Field(
        default=2, description="The upscale factor"
    )
    guidance_scale: float = Field(
        default=4, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=18, description="The number of inference steps to perform."
    )
    seed: str = Field(
        default="", description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )
    negative_prompt: str = Field(
        default="(worst quality, low quality, normal quality:2)", description="The negative prompt to use. Use it to address details that you don't want in the image."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to false, the safety checker will be disabled."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "resemblance": self.resemblance,
            "creativity": self.creativity,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "upscale_factor": self.upscale_factor,
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
            application="fal-ai/clarity-upscaler",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["image", "scale"]

class RecraftV3ImageToImage(FALNode):
    """
    Recraft V3 transforms images with advanced style control and quality preservation. Professional-grade image-to-image generation with fine-tuned artistic control.
    image, transformation, recraft, style, professional

    Use cases:
    - Transform images with precise style control
    - Create high-quality image variations
    - Apply artistic modifications with consistency
    - Generate professional design alternatives
    - Produce style-coherent image transformations
    """

    class RecraftV3ImageToImageStyle(Enum):
        """
        The style of the generated images. Vector images cost 2X as much.
        """
        ANY = "any"
        REALISTIC_IMAGE = "realistic_image"
        DIGITAL_ILLUSTRATION = "digital_illustration"
        VECTOR_ILLUSTRATION = "vector_illustration"
        REALISTIC_IMAGE__B_AND_W = "realistic_image/b_and_w"
        REALISTIC_IMAGE__HARD_FLASH = "realistic_image/hard_flash"
        REALISTIC_IMAGE__HDR = "realistic_image/hdr"
        REALISTIC_IMAGE__NATURAL_LIGHT = "realistic_image/natural_light"
        REALISTIC_IMAGE__STUDIO_PORTRAIT = "realistic_image/studio_portrait"
        REALISTIC_IMAGE__ENTERPRISE = "realistic_image/enterprise"
        REALISTIC_IMAGE__MOTION_BLUR = "realistic_image/motion_blur"
        REALISTIC_IMAGE__EVENING_LIGHT = "realistic_image/evening_light"
        REALISTIC_IMAGE__FADED_NOSTALGIA = "realistic_image/faded_nostalgia"
        REALISTIC_IMAGE__FOREST_LIFE = "realistic_image/forest_life"
        REALISTIC_IMAGE__MYSTIC_NATURALISM = "realistic_image/mystic_naturalism"
        REALISTIC_IMAGE__NATURAL_TONES = "realistic_image/natural_tones"
        REALISTIC_IMAGE__ORGANIC_CALM = "realistic_image/organic_calm"
        REALISTIC_IMAGE__REAL_LIFE_GLOW = "realistic_image/real_life_glow"
        REALISTIC_IMAGE__RETRO_REALISM = "realistic_image/retro_realism"
        REALISTIC_IMAGE__RETRO_SNAPSHOT = "realistic_image/retro_snapshot"
        REALISTIC_IMAGE__URBAN_DRAMA = "realistic_image/urban_drama"
        REALISTIC_IMAGE__VILLAGE_REALISM = "realistic_image/village_realism"
        REALISTIC_IMAGE__WARM_FOLK = "realistic_image/warm_folk"
        DIGITAL_ILLUSTRATION__PIXEL_ART = "digital_illustration/pixel_art"
        DIGITAL_ILLUSTRATION__HAND_DRAWN = "digital_illustration/hand_drawn"
        DIGITAL_ILLUSTRATION__GRAIN = "digital_illustration/grain"
        DIGITAL_ILLUSTRATION__INFANTILE_SKETCH = "digital_illustration/infantile_sketch"
        DIGITAL_ILLUSTRATION__2D_ART_POSTER = "digital_illustration/2d_art_poster"
        DIGITAL_ILLUSTRATION__HANDMADE_3D = "digital_illustration/handmade_3d"
        DIGITAL_ILLUSTRATION__HAND_DRAWN_OUTLINE = "digital_illustration/hand_drawn_outline"
        DIGITAL_ILLUSTRATION__ENGRAVING_COLOR = "digital_illustration/engraving_color"
        DIGITAL_ILLUSTRATION__2D_ART_POSTER_2 = "digital_illustration/2d_art_poster_2"
        DIGITAL_ILLUSTRATION__ANTIQUARIAN = "digital_illustration/antiquarian"
        DIGITAL_ILLUSTRATION__BOLD_FANTASY = "digital_illustration/bold_fantasy"
        DIGITAL_ILLUSTRATION__CHILD_BOOK = "digital_illustration/child_book"
        DIGITAL_ILLUSTRATION__CHILD_BOOKS = "digital_illustration/child_books"
        DIGITAL_ILLUSTRATION__COVER = "digital_illustration/cover"
        DIGITAL_ILLUSTRATION__CROSSHATCH = "digital_illustration/crosshatch"
        DIGITAL_ILLUSTRATION__DIGITAL_ENGRAVING = "digital_illustration/digital_engraving"
        DIGITAL_ILLUSTRATION__EXPRESSIONISM = "digital_illustration/expressionism"
        DIGITAL_ILLUSTRATION__FREEHAND_DETAILS = "digital_illustration/freehand_details"
        DIGITAL_ILLUSTRATION__GRAIN_20 = "digital_illustration/grain_20"
        DIGITAL_ILLUSTRATION__GRAPHIC_INTENSITY = "digital_illustration/graphic_intensity"
        DIGITAL_ILLUSTRATION__HARD_COMICS = "digital_illustration/hard_comics"
        DIGITAL_ILLUSTRATION__LONG_SHADOW = "digital_illustration/long_shadow"
        DIGITAL_ILLUSTRATION__MODERN_FOLK = "digital_illustration/modern_folk"
        DIGITAL_ILLUSTRATION__MULTICOLOR = "digital_illustration/multicolor"
        DIGITAL_ILLUSTRATION__NEON_CALM = "digital_illustration/neon_calm"
        DIGITAL_ILLUSTRATION__NOIR = "digital_illustration/noir"
        DIGITAL_ILLUSTRATION__NOSTALGIC_PASTEL = "digital_illustration/nostalgic_pastel"
        DIGITAL_ILLUSTRATION__OUTLINE_DETAILS = "digital_illustration/outline_details"
        DIGITAL_ILLUSTRATION__PASTEL_GRADIENT = "digital_illustration/pastel_gradient"
        DIGITAL_ILLUSTRATION__PASTEL_SKETCH = "digital_illustration/pastel_sketch"
        DIGITAL_ILLUSTRATION__POP_ART = "digital_illustration/pop_art"
        DIGITAL_ILLUSTRATION__POP_RENAISSANCE = "digital_illustration/pop_renaissance"
        DIGITAL_ILLUSTRATION__STREET_ART = "digital_illustration/street_art"
        DIGITAL_ILLUSTRATION__TABLET_SKETCH = "digital_illustration/tablet_sketch"
        DIGITAL_ILLUSTRATION__URBAN_GLOW = "digital_illustration/urban_glow"
        DIGITAL_ILLUSTRATION__URBAN_SKETCHING = "digital_illustration/urban_sketching"
        DIGITAL_ILLUSTRATION__VANILLA_DREAMS = "digital_illustration/vanilla_dreams"
        DIGITAL_ILLUSTRATION__YOUNG_ADULT_BOOK = "digital_illustration/young_adult_book"
        DIGITAL_ILLUSTRATION__YOUNG_ADULT_BOOK_2 = "digital_illustration/young_adult_book_2"
        VECTOR_ILLUSTRATION__BOLD_STROKE = "vector_illustration/bold_stroke"
        VECTOR_ILLUSTRATION__CHEMISTRY = "vector_illustration/chemistry"
        VECTOR_ILLUSTRATION__COLORED_STENCIL = "vector_illustration/colored_stencil"
        VECTOR_ILLUSTRATION__CONTOUR_POP_ART = "vector_illustration/contour_pop_art"
        VECTOR_ILLUSTRATION__COSMICS = "vector_illustration/cosmics"
        VECTOR_ILLUSTRATION__CUTOUT = "vector_illustration/cutout"
        VECTOR_ILLUSTRATION__DEPRESSIVE = "vector_illustration/depressive"
        VECTOR_ILLUSTRATION__EDITORIAL = "vector_illustration/editorial"
        VECTOR_ILLUSTRATION__EMOTIONAL_FLAT = "vector_illustration/emotional_flat"
        VECTOR_ILLUSTRATION__INFOGRAPHICAL = "vector_illustration/infographical"
        VECTOR_ILLUSTRATION__MARKER_OUTLINE = "vector_illustration/marker_outline"
        VECTOR_ILLUSTRATION__MOSAIC = "vector_illustration/mosaic"
        VECTOR_ILLUSTRATION__NAIVECTOR = "vector_illustration/naivector"
        VECTOR_ILLUSTRATION__ROUNDISH_FLAT = "vector_illustration/roundish_flat"
        VECTOR_ILLUSTRATION__SEGMENTED_COLORS = "vector_illustration/segmented_colors"
        VECTOR_ILLUSTRATION__SHARP_CONTRAST = "vector_illustration/sharp_contrast"
        VECTOR_ILLUSTRATION__THIN = "vector_illustration/thin"
        VECTOR_ILLUSTRATION__VECTOR_PHOTO = "vector_illustration/vector_photo"
        VECTOR_ILLUSTRATION__VIVID_SHAPES = "vector_illustration/vivid_shapes"
        VECTOR_ILLUSTRATION__ENGRAVING = "vector_illustration/engraving"
        VECTOR_ILLUSTRATION__LINE_ART = "vector_illustration/line_art"
        VECTOR_ILLUSTRATION__LINE_CIRCUIT = "vector_illustration/line_circuit"
        VECTOR_ILLUSTRATION__LINOCUT = "vector_illustration/linocut"


    prompt: str = Field(
        default="", description="The text prompt describing the desired transformation"
    )
    style: RecraftV3ImageToImageStyle = Field(
        default=RecraftV3ImageToImageStyle.REALISTIC_IMAGE, description="The artistic style to apply"
    )
    style_id: str = Field(
        default="", description="The ID of the custom style reference (optional)"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to modify. Must be less than 5 MB in size, have resolution less than 16 MP and max dimension less than 4096 pixels."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    strength: float = Field(
        default=0.5, description="Defines the difference with the original image, should lie in [0, 1], where 0 means almost identical, and 1 means miserable similarity"
    )
    colors: list[str] = Field(
        default=[], description="An array of preferable colors"
    )
    negative_prompt: str = Field(
        default="", description="A text description of undesired elements on an image"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "style": self.style.value,
            "style_id": self.style_id,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "strength": self.strength,
            "colors": self.colors,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/recraft/v3/image-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image", "style"]

class KolorsImageToImage(FALNode):
    """
    Kolors transforms images using an advanced diffusion model. High-quality image-to-image generation with natural color preservation and detail retention.
    image, transformation, kolors, diffusion, quality

    Use cases:
    - Transform images with natural color handling
    - Create variations with preserved color harmony
    - Apply modifications with detail retention
    - Generate style transfers with color consistency
    - Produce high-fidelity image transformations
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
        default="", description="The text prompt describing the desired transformation"
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="", description="The size of the generated image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of image to use for image to image"
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the image to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the image directly in the response without going through the CDN."
    )
    scheduler: Scheduler = Field(
        default=Scheduler.EULERDISCRETESCHEDULER, description="The scheduler to use for the model."
    )
    strength: float = Field(
        default=0.85, description="Strength of the transformation (0-1, higher = more change)"
    )
    guidance_scale: float = Field(
        default=5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=50, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="Seed for reproducible results. Use -1 for random"
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Enable safety checker."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "scheduler": self.scheduler.value,
            "strength": self.strength,
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
            application="fal-ai/kolors/image-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image", "strength"]

class BiRefNet(FALNode):
    """
    BiRefNet (Bilateral Reference Network) performs high-quality background removal with precise edge detection and detail preservation.
    image, background-removal, segmentation, birefnet, mask

    Use cases:
    - Remove backgrounds from product photos
    - Create transparent PNGs from images
    - Extract subjects for compositing
    - Generate clean cutouts for design work
    - Prepare images for background replacement
    """

    class OperatingResolution(Enum):
        """
        The resolution to operate on. The higher the resolution, the more accurate the output will be for high res input images.
        """
        VALUE_1024X1024 = "1024x1024"
        VALUE_2048X2048 = "2048x2048"

    class OutputFormat(Enum):
        """
        The format of the output image
        """
        WEBP = "webp"
        PNG = "png"
        GIF = "gif"

    class Model(Enum):
        """
        Model to use for background removal.
        The 'General Use (Light)' model is the original model used in the BiRefNet repository.
        The 'General Use (Heavy)' model is a slower but more accurate model.
        The 'Portrait' model is a model trained specifically for portrait images.
        The 'General Use (Light)' model is recommended for most use cases.
        The corresponding models are as follows:
        - 'General Use (Light)': BiRefNet-DIS_ep580.pth
        - 'General Use (Heavy)': BiRefNet-massive-epoch_240.pth
        - 'Portrait': BiRefNet-portrait-TR_P3M_10k-epoch_120.pth
        """
        GENERAL_USE_LIGHT = "General Use (Light)"
        GENERAL_USE_HEAVY = "General Use (Heavy)"
        PORTRAIT = "Portrait"


    operating_resolution: OperatingResolution = Field(
        default=OperatingResolution.VALUE_1024X1024, description="The resolution to operate on. The higher the resolution, the more accurate the output will be for high res input images."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image to remove background from"
    )
    model: Model = Field(
        default=Model.GENERAL_USE_LIGHT, description="Model to use for background removal. The 'General Use (Light)' model is the original model used in the BiRefNet repository. The 'General Use (Heavy)' model is a slower but more accurate model. The 'Portrait' model is a model trained specifically for portrait images. The 'General Use (Light)' model is recommended for most use cases. The corresponding models are as follows: - 'General Use (Light)': BiRefNet-DIS_ep580.pth - 'General Use (Heavy)': BiRefNet-massive-epoch_240.pth - 'Portrait': BiRefNet-portrait-TR_P3M_10k-epoch_120.pth"
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

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "operating_resolution": self.operating_resolution.value,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "model": self.model.value,
            "sync_mode": self.sync_mode,
            "output_mask": self.output_mask,
            "refine_foreground": self.refine_foreground,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/birefnet",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["image"]

class CodeFormer(FALNode):
    """
    CodeFormer restores and enhances face quality in images. Advanced face restoration with fidelity control for natural-looking results.
    image, face-restoration, enhancement, codeformer, quality

    Use cases:
    - Restore quality in degraded face photos
    - Enhance facial details in low-quality images
    - Improve portrait quality for professional use
    - Fix compressed or damaged face images
    - Enhance facial features while maintaining identity
    """

    aligned: bool = Field(
        default=False, description="Should faces etc should be aligned."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of image to be used for relighting"
    )
    upscale_factor: float = Field(
        default=2, description="Upscaling factor"
    )
    fidelity: float = Field(
        default=0.5, description="Fidelity level (0-1, higher = more faithful to input)"
    )
    face_upscale: bool = Field(
        default=True, description="Should faces be upscaled"
    )
    only_center_face: bool = Field(
        default=False, description="Should only center face be restored"
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducible generation."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "aligned": self.aligned,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "upscale_factor": self.upscale_factor,
            "fidelity": self.fidelity,
            "face_upscale": self.face_upscale,
            "only_center_face": self.only_center_face,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/codeformer",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["image", "fidelity"]

class HunyuanImageV3InstructEdit(FALNode):
    """
    Hunyuan Image v3 Instruct Edit allows precise image editing through natural language instructions with advanced understanding.
    image, editing, hunyuan, instruct, ai-editing

    Use cases:
    - Edit images using natural language instructions
    - Modify specific elements in photos with text commands
    - Apply precise adjustments through conversational editing
    - Transform images with instruction-based control
    - Create variations with detailed text guidance
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
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images to use as a reference for the generation. A maximum of 2 images are supported."
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
            "image_urls": self.image_urls,
            "guidance_scale": self.guidance_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan-image/v3/instruct/edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageMaxEdit(FALNode):
    """
    Qwen Image Max Edit provides powerful image editing capabilities with advanced AI understanding and high-quality results.
    image, editing, qwen, max, ai-editing

    Use cases:
    - Edit images with advanced AI understanding
    - Apply complex modifications to photos
    - Transform images with high-quality results
    - Create professional edits with natural prompts
    - Modify images with precise control
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
        default="", description="The size of the generated image. If not provided, the size of the final input image will be used."
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
    image_urls: list[str] = Field(
        default=[], description="Reference images for editing (1-3 images required). Order matters: reference as 'image 1', 'image 2', 'image 3' in prompt. Resolution: 384-5000px each dimension. Max size: 10MB each. Formats: JPEG, JPG, PNG (no alpha), WEBP."
    )
    negative_prompt: str = Field(
        default="", description="Content to avoid in the generated image. Max 500 characters."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Enable content moderation for input and output."
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
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-max/edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEdit2511(FALNode):
    """
    Qwen Image Edit 2511 provides state-of-the-art image editing with latest AI advancements and improved quality.
    image, editing, qwen, 2511, latest

    Use cases:
    - Edit images with latest Qwen technology
    - Apply advanced modifications to photos
    - Create high-quality edits with AI assistance
    - Transform images with cutting-edge models
    - Produce professional image modifications
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
        default="", description="The prompt to edit the image with."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If None, uses the input image dimensions."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI."
    )
    guidance_scale: float = Field(
        default=4.5, description="The guidance scale to use for the image generation."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images to edit."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to generate an image from."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
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
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-2511",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEdit2511Lora(FALNode):
    """
    Qwen Image Edit 2511 with LoRA support enables custom-trained models for specialized editing tasks.
    image, editing, qwen, lora, custom

    Use cases:
    - Edit images with custom-trained models
    - Apply specialized modifications using LoRA
    - Create domain-specific edits
    - Transform images with fine-tuned models
    - Produce customized image modifications
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
        default="", description="The prompt to edit the image with."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If None, uses the input image dimensions."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    loras: list[str] = Field(
        default=[], description="The LoRAs to use for the image generation. You can use up to 3 LoRAs and they will be merged together to generate the final image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI."
    )
    guidance_scale: float = Field(
        default=4.5, description="The guidance scale to use for the image generation."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images to edit."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to generate an image from."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "acceleration": self.acceleration.value,
            "num_inference_steps": self.num_inference_steps,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "loras": self.loras,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-2511/lora",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEdit2511MultipleAngles(FALNode):
    """
    Qwen Image Edit 2511 Multiple Angles generates images from different viewpoints based on a single input image.
    image, editing, qwen, multi-angle, viewpoint

    Use cases:
    - Generate multiple viewpoints from single image
    - Create product views from different angles
    - Visualize objects from various perspectives
    - Produce multi-angle image sets
    - Transform images to show different sides
    """

    class Acceleration(Enum):
        """
        Acceleration level for image generation.
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


    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation."
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the input image will be used."
    )
    horizontal_angle: float = Field(
        default=0, description="Horizontal rotation angle around the object in degrees. 0°=front view, 90°=right side, 180°=back view, 270°=left side, 360°=front view again."
    )
    guidance_scale: float = Field(
        default=4.5, description="The CFG (Classifier Free Guidance) scale."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    image_urls: list[str] = Field(
        default=[], description="The URL of the image to adjust camera angle for."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt for the generation"
    )
    zoom: float = Field(
        default=5, description="Camera zoom/distance. 0=wide shot (far away), 5=medium shot (normal), 10=close-up (very close)."
    )
    vertical_angle: float = Field(
        default=0, description="Vertical camera angle in degrees. -30°=low-angle shot (looking up), 0°=eye-level, 30°=elevated, 60°=high-angle, 90°=bird's-eye view (looking down)."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    lora_scale: float = Field(
        default=1, description="The scale factor for the LoRA model. Controls the strength of the camera control effect."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI."
    )
    additional_prompt: str = Field(
        default="", description="Additional text to append to the automatically generated prompt."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "acceleration": self.acceleration.value,
            "image_size": self.image_size,
            "horizontal_angle": self.horizontal_angle,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "zoom": self.zoom,
            "vertical_angle": self.vertical_angle,
            "num_images": self.num_images,
            "lora_scale": self.lora_scale,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "additional_prompt": self.additional_prompt,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-2511-multiple-angles",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEdit2509(FALNode):
    """
    Qwen Image Edit 2509 provides powerful image editing with advanced AI capabilities and high-quality output.
    image, editing, qwen, 2509, ai-editing

    Use cases:
    - Edit images with Qwen 2509 technology
    - Apply sophisticated modifications to photos
    - Create quality edits with AI assistance
    - Transform images with advanced models
    - Produce professional image changes
    """

    class Acceleration(Enum):
        """
        Acceleration level for image generation. Options: 'none', 'regular'. Higher acceleration increases speed. 'regular' balances speed and quality.
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
        default="", description="The prompt to generate the image with"
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="square_hd", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. Options: 'none', 'regular'. Higher acceleration increases speed. 'regular' balances speed and quality."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_scale: float = Field(
        default=4, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images to edit."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    num_inference_steps: int = Field(
        default=50, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-2509",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEdit2509Lora(FALNode):
    """
    Qwen Image Edit 2509 with LoRA enables fine-tuned models for specialized image editing applications.
    image, editing, qwen, lora, fine-tuned

    Use cases:
    - Edit images with fine-tuned models
    - Apply custom modifications using LoRA
    - Create specialized edits for specific domains
    - Transform images with trained models
    - Produce tailored image modifications
    """

    class Acceleration(Enum):
        """
        Acceleration level for image generation. Options: 'none', 'regular'. Higher acceleration increases speed. 'regular' balances speed and quality.
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
        default="", description="The prompt to generate the image with"
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the final input image will be used to calculate the size of the output image."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. Options: 'none', 'regular'. Higher acceleration increases speed. 'regular' balances speed and quality."
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
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images to edit."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    guidance_scale: float = Field(
        default=4, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "seed": self.seed,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "loras": self.loras,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "guidance_scale": self.guidance_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-2509-lora",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageLayered(FALNode):
    """
    Qwen Image Layered provides layer-based image editing for complex compositions and precise control.
    image, editing, qwen, layered, composition

    Use cases:
    - Edit images with layer-based control
    - Create complex compositions
    - Apply modifications to specific layers
    - Build multi-layer image edits
    - Produce sophisticated image compositions
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
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="A caption for the input image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    num_layers: int = Field(
        default=4, description="The number of layers to generate."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the input image."
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
        default=5, description="The guidance scale to use for the image generation."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to generate an image from."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "acceleration": self.acceleration.value,
            "num_layers": self.num_layers,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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
            application="fal-ai/qwen-image-layered",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageLayeredLora(FALNode):
    """
    Qwen Image Layered with LoRA combines layer-based editing with custom-trained models for specialized tasks.
    image, editing, qwen, layered, lora

    Use cases:
    - Edit layered images with custom models
    - Create specialized layer compositions
    - Apply fine-tuned modifications
    - Build complex edits with trained models
    - Produce custom layer-based results
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
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="A caption for the input image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    num_layers: int = Field(
        default=4, description="The number of layers to generate."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the input image."
    )
    loras: list[str] = Field(
        default=[], description="List of LoRA weights to apply (maximum 3)."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_scale: float = Field(
        default=5, description="The guidance scale to use for the image generation."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
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
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "acceleration": self.acceleration.value,
            "num_layers": self.num_layers,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "loras": self.loras,
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
            application="fal-ai/qwen-image-layered/lora",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Flux2Klein4BBaseEdit(FALNode):
    """
    FLUX-2 Klein 4B Base Edit provides fast image editing with the 4-billion parameter model.
    image, editing, flux-2, klein, 4b

    Use cases:
    - Edit images with FLUX-2 Klein 4B
    - Apply fast modifications to photos
    - Create quick edits with AI assistance
    - Transform images efficiently
    - Produce rapid image modifications
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
        default="", description="The prompt to edit the image."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, uses the input image size."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use for image generation."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for classifier-free guidance."
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
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images for editing. A maximum of 4 images are allowed."
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
            "guidance_scale": self.guidance_scale,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2/klein/4b/base/edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Flux2Klein4BBaseEditLora(FALNode):
    """
    FLUX-2 Klein 4B Base Edit with LoRA enables custom-trained models for specialized editing.
    image, editing, flux-2, klein, 4b, lora

    Use cases:
    - Edit images with custom FLUX-2 models
    - Apply specialized modifications using LoRA
    - Create domain-specific edits
    - Transform images with fine-tuned 4B model
    - Produce customized modifications
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
        default="", description="The prompt to edit the image."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, uses the input image size."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use for image generation."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for classifier-free guidance."
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
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images for editing. A maximum of 4 images are allowed."
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
            "guidance_scale": self.guidance_scale,
            "output_format": self.output_format.value,
            "loras": self.loras,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2/klein/4b/base/edit/lora",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Flux2Klein9BBaseEdit(FALNode):
    """
    FLUX-2 Klein 9B Base Edit provides high-quality image editing with the 9-billion parameter model.
    image, editing, flux-2, klein, 9b

    Use cases:
    - Edit images with FLUX-2 Klein 9B
    - Apply high-quality modifications to photos
    - Create advanced edits with powerful AI
    - Transform images with superior quality
    - Produce professional image modifications
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
        default="", description="The prompt to edit the image."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, uses the input image size."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use for image generation."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for classifier-free guidance."
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
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images for editing. A maximum of 4 images are allowed."
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
            "guidance_scale": self.guidance_scale,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2/klein/9b/base/edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Flux2Klein9BBaseEditLora(FALNode):
    """
    FLUX-2 Klein 9B Base Edit with LoRA combines powerful editing with custom-trained models.
    image, editing, flux-2, klein, 9b, lora

    Use cases:
    - Edit images with custom 9B models
    - Apply specialized high-quality modifications
    - Create professional custom edits
    - Transform images with fine-tuned powerful model
    - Produce advanced customized results
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
        default="", description="The prompt to edit the image."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, uses the input image size."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use for image generation."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for classifier-free guidance."
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
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images for editing. A maximum of 4 images are allowed."
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
            "guidance_scale": self.guidance_scale,
            "output_format": self.output_format.value,
            "loras": self.loras,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2/klein/9b/base/edit/lora",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Flux2Klein4BEdit(FALNode):
    """
    FLUX-2 Klein 4B Edit provides efficient image editing with the streamlined 4-billion parameter model.
    image, editing, flux-2, klein, 4b, efficient

    Use cases:
    - Edit images efficiently with FLUX-2
    - Apply quick modifications to photos
    - Create fast edits for rapid workflows
    - Transform images with streamlined model
    - Produce quick image modifications
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to edit the image."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, uses the input image size."
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
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images for editing. A maximum of 4 images are allowed."
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
            "image_urls": self.image_urls,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2/klein/4b/edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Flux2Klein9BEdit(FALNode):
    """
    FLUX-2 Klein 9B Edit provides advanced image editing with the full 9-billion parameter model.
    image, editing, flux-2, klein, 9b, advanced

    Use cases:
    - Edit images with advanced FLUX-2 model
    - Apply sophisticated modifications
    - Create high-quality edits
    - Transform images with powerful AI
    - Produce superior image modifications
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to edit the image."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, uses the input image size."
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
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images for editing. A maximum of 4 images are allowed."
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
            "image_urls": self.image_urls,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2/klein/9b/edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Flux2FlashEdit(FALNode):
    """
    FLUX-2 Flash Edit provides ultra-fast image editing for rapid iteration and quick modifications.
    image, editing, flux-2, flash, ultra-fast

    Use cases:
    - Edit images with ultra-fast processing
    - Apply instant modifications to photos
    - Create rapid edits for quick turnaround
    - Transform images at maximum speed
    - Produce instant image modifications
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to edit the image."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="square_hd", description="The size of the image to generate. The width and height must be between 512 and 2048 pixels."
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
        default=-1, description="The seed to use for the generation. If not provided, a random seed will be used."
    )
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images for editing. A maximum of 4 images are allowed, if more are provided, only the first 4 will be used."
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
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2/flash/edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Flux2TurboEdit(FALNode):
    """
    FLUX-2 Turbo Edit provides accelerated image editing with balanced quality and speed.
    image, editing, flux-2, turbo, fast

    Use cases:
    - Edit images with turbo speed
    - Apply fast modifications with good quality
    - Create quick edits efficiently
    - Transform images rapidly
    - Produce fast quality modifications
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to edit the image."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="square_hd", description="The size of the image to generate. The width and height must be between 512 and 2048 pixels."
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
        default=-1, description="The seed to use for the generation. If not provided, a random seed will be used."
    )
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images for editing. A maximum of 4 images are allowed, if more are provided, only the first 4 will be used."
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
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2/turbo/edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Flux2MaxEdit(FALNode):
    """
    FLUX-2 Max Edit provides maximum quality image editing with the most advanced FLUX-2 model.
    image, editing, flux-2, max, premium

    Use cases:
    - Edit images with maximum quality
    - Apply premium modifications to photos
    - Create professional-grade edits
    - Transform images with best quality
    - Produce highest quality modifications
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
        default="auto", description="The size of the generated image. If `auto`, the size will be determined by the model."
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
    image_urls: list[str] = Field(
        default=[], description="List of URLs of input images for editing"
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
            "image_urls": self.image_urls,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2-max/edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Flux2FlexEdit(FALNode):
    """
    FLUX-2 Flex Edit provides flexible image editing with customizable parameters and versatile control.
    image, editing, flux-2, flex, versatile

    Use cases:
    - Edit images with flexible controls
    - Apply customizable modifications
    - Create versatile edits
    - Transform images with adaptable settings
    - Produce flexible image modifications
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
    guidance_scale: float = Field(
        default=3.5, description="The guidance scale to use for the generation."
    )
    image_size: str = Field(
        default="auto", description="The size of the generated image. If `auto`, the size will be determined by the model."
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
    image_urls: list[str] = Field(
        default=[], description="List of URLs of input images for editing"
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
            "guidance_scale": self.guidance_scale,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "safety_tolerance": self.safety_tolerance.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "num_inference_steps": self.num_inference_steps,
            "image_urls": self.image_urls,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2-flex/edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Flux2LoraGalleryVirtualTryon(FALNode):
    """
    FLUX-2 LoRA Gallery Virtual Try-on enables realistic clothing and accessory visualization on people.
    image, editing, flux-2, virtual-tryon, fashion

    Use cases:
    - Visualize clothing on models
    - Try on accessories virtually
    - Create fashion previews
    - Test product appearances
    - Generate try-on images
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
        default="", description="The prompt to generate a virtual try-on image."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the input image will be used."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    lora_scale: float = Field(
        default=1, description="The strength of the virtual try-on effect."
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
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images for virtual try-on. Provide person image and clothing image."
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
            "image_urls": self.image_urls,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2-lora-gallery/virtual-tryon",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Flux2LoraGalleryMultipleAngles(FALNode):
    """
    FLUX-2 LoRA Gallery Multiple Angles generates images from different viewpoints for comprehensive visualization.
    image, editing, flux-2, multi-angle, viewpoint

    Use cases:
    - Generate multiple product angles
    - Create viewpoint variations
    - Visualize objects from different sides
    - Produce multi-angle image sets
    - Generate comprehensive views
    """

    class Acceleration(Enum):
        """
        Acceleration level for image generation.
        """
        NONE = "none"
        REGULAR = "regular"

    class OutputFormat(Enum):
        """
        The format of the output image.
        """
        PNG = "png"
        JPEG = "jpeg"
        WEBP = "webp"


    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the input image will be used."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation."
    )
    horizontal_angle: float = Field(
        default=0, description="Horizontal rotation angle around the object in degrees. 0°=front view, 90°=right side, 180°=back view, 270°=left side, 360°=front view again."
    )
    guidance_scale: float = Field(
        default=2.5, description="The CFG (Classifier Free Guidance) scale."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    image_urls: list[str] = Field(
        default=[], description="The URL of the image to adjust camera angle for."
    )
    zoom: float = Field(
        default=5, description="Camera zoom/distance. 0=wide shot (far away), 5=medium shot (normal), 10=close-up (very close)."
    )
    vertical_angle: float = Field(
        default=0, description="Vertical camera angle in degrees. 0°=eye-level shot, 30°=elevated shot, 60°=high-angle shot (looking down from above)."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate."
    )
    lora_scale: float = Field(
        default=1, description="The strength of the multiple angles effect."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image."
    )
    sync_mode: bool = Field(
        default=False, description="If True, the media will be returned as a data URI."
    )
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to perform."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "horizontal_angle": self.horizontal_angle,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "image_urls": self.image_urls,
            "zoom": self.zoom,
            "vertical_angle": self.vertical_angle,
            "num_images": self.num_images,
            "lora_scale": self.lora_scale,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2-lora-gallery/multiple-angles",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Flux2LoraGalleryFaceToFullPortrait(FALNode):
    """
    FLUX-2 LoRA Gallery Face to Full Portrait expands face crops into complete portrait images.
    image, editing, flux-2, portrait, expansion

    Use cases:
    - Expand face crops to full portraits
    - Generate complete portrait from face
    - Create full-body images from headshots
    - Extend facial images to portraits
    - Produce complete portrait compositions
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
        default="Face to full portrait", description="The prompt describing the full portrait to generate from the face."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the input image will be used."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    lora_scale: float = Field(
        default=1, description="The strength of the face to full portrait effect."
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
    image_urls: list[str] = Field(
        default=[], description="The URL of the cropped face image."
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
            "image_urls": self.image_urls,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2-lora-gallery/face-to-full-portrait",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Flux2LoraGalleryAddBackground(FALNode):
    """
    FLUX-2 LoRA Gallery Add Background places subjects in new environments with realistic integration.
    image, editing, flux-2, background, compositing

    Use cases:
    - Add backgrounds to cutout images
    - Place subjects in new environments
    - Create realistic background compositions
    - Generate contextual settings
    - Produce integrated background images
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
        default="Add Background forest", description="The prompt describing the background to add. Must start with 'Add Background' followed by your description."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the input image will be used."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    lora_scale: float = Field(
        default=1, description="The strength of the add background effect."
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
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images. Provide an image with a white or clean background."
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
            "image_urls": self.image_urls,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2-lora-gallery/add-background",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class BriaFiboEdit(FALNode):
    """
    Bria FIBO Edit provides general-purpose image editing with AI-powered modifications and enhancements.
    image, editing, bria, fibo, general

    Use cases:
    - Edit images with general-purpose AI
    - Apply various modifications to photos
    - Create edited versions of images
    - Transform images with flexible edits
    - Produce AI-powered modifications
    """

    steps_num: int = Field(
        default=50, description="Number of inference steps."
    )
    instruction: str = Field(
        default="", description="Instruction for image editing."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="Reference image (file or URL)."
    )
    sync_mode: bool = Field(
        default=False, description="If true, returns the image directly in the response (increases latency)."
    )
    guidance_scale: str = Field(
        default=5, description="Guidance scale for text."
    )
    structured_instruction: str = Field(
        default="", description="The structured prompt to generate an image from."
    )
    mask_url: ImageRef = Field(
        default=ImageRef(), description="Mask image (file or URL). Optional"
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt for image generation."
    )
    seed: int = Field(
        default=5555, description="Random seed for reproducibility."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        mask_url_base64 = await context.image_to_base64(self.mask_url)
        arguments = {
            "steps_num": self.steps_num,
            "instruction": self.instruction,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "structured_instruction": self.structured_instruction,
            "mask_url": f"data:image/png;base64,{mask_url_base64}",
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/fibo-edit/edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class BriaFiboEditAddObjectByText(FALNode):
    """
    Bria FIBO Edit Add Object by Text inserts new objects into images using text descriptions.
    image, editing, bria, fibo, object-insertion

    Use cases:
    - Add objects to images with text
    - Insert elements using descriptions
    - Place new items in scenes
    - Augment images with additional objects
    - Generate object additions
    """

    instruction: str = Field(
        default="", description="The full natural language command describing what to add and where."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The source image."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "instruction": self.instruction,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/fibo-edit/add_object_by_text",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class BriaFiboEditEraseByText(FALNode):
    """
    Bria FIBO Edit Erase by Text removes objects from images using natural language descriptions.
    image, editing, bria, fibo, object-removal

    Use cases:
    - Remove objects using text descriptions
    - Erase unwanted elements from photos
    - Clean up images by describing what to remove
    - Delete specific items from scenes
    - Remove objects with natural language
    """

    object_name: str = Field(
        default="", description="The name of the object to remove."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The source image."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "object_name": self.object_name,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/fibo-edit/erase_by_text",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class BriaFiboEditReplaceObjectByText(FALNode):
    """
    Bria FIBO Edit Replace Object by Text replaces objects in images with new ones specified by text.
    image, editing, bria, fibo, object-replacement

    Use cases:
    - Replace objects using text descriptions
    - Swap elements in photos
    - Change specific items in scenes
    - Transform objects with text guidance
    - Substitute objects with new ones
    """

    instruction: str = Field(
        default="", description="The full natural language command describing what to replace."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The source image."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "instruction": self.instruction,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/fibo-edit/replace_object_by_text",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class BriaFiboEditBlend(FALNode):
    """
    Bria FIBO Edit Blend seamlessly combines multiple images or elements with natural transitions.
    image, editing, bria, fibo, blending

    Use cases:
    - Blend multiple images together
    - Create seamless compositions
    - Merge elements naturally
    - Combine images with smooth transitions
    - Generate blended composites
    """

    instruction: str = Field(
        default="", description="Instruct what elements you would like to blend in your image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The source image."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "instruction": self.instruction,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/fibo-edit/blend",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class BriaFiboEditColorize(FALNode):
    """
    Bria FIBO Edit Colorize adds realistic colors to grayscale or black-and-white images.
    image, editing, bria, fibo, colorization

    Use cases:
    - Colorize black and white photos
    - Add colors to grayscale images
    - Restore color in old photographs
    - Transform monochrome to color
    - Generate colored versions of grayscale images
    """

    class Color(Enum):
        """
        Select the color palette or aesthetic for the output image
        """
        CONTEMPORARY_COLOR = "contemporary color"
        VIVID_COLOR = "vivid color"
        BLACK_AND_WHITE_COLORS = "black and white colors"
        SEPIA_VINTAGE = "sepia vintage"


    color: Color = Field(
        default="", description="Select the color palette or aesthetic for the output image"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The source image."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "color": self.color.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/fibo-edit/colorize",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image"]

class BriaFiboEditRestore(FALNode):
    """
    Bria FIBO Edit Restore repairs and enhances damaged or degraded images with AI reconstruction.
    image, editing, bria, fibo, restoration

    Use cases:
    - Restore damaged photographs
    - Repair degraded images
    - Enhance old photo quality
    - Fix scratches and artifacts
    - Reconstruct missing image parts
    """

    image_url: ImageRef = Field(
        default=ImageRef(), description="The source image."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/fibo-edit/restore",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image"]

class BriaFiboEditRestyle(FALNode):
    """
    Bria FIBO Edit Restyle transforms images with artistic style transfers and visual aesthetics.
    image, editing, bria, fibo, style-transfer

    Use cases:
    - Apply artistic styles to images
    - Transform photos with new aesthetics
    - Create stylized versions of images
    - Generate artistic variations
    - Produce style-transferred images
    """

    class Style(Enum):
        """
        Select the desired artistic style for the output image.
        """
        RENDER_3D = "3D Render"
        CUBISM = "Cubism"
        OIL_PAINTING = "Oil Painting"
        ANIME = "Anime"
        CARTOON = "Cartoon"
        COLORING_BOOK = "Coloring Book"
        RETRO_AD = "Retro Ad"
        POP_ART_HALFTONE = "Pop Art Halftone"
        VECTOR_ART = "Vector Art"
        STORY_BOARD = "Story Board"
        ART_NOUVEAU = "Art Nouveau"
        CROSS_ETCHING = "Cross Etching"
        WOOD_CUT = "Wood Cut"


    style: Style = Field(
        default="", description="Select the desired artistic style for the output image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The source image."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "style": self.style.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/fibo-edit/restyle",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class BriaFiboEditRelight(FALNode):
    """
    Bria FIBO Edit Relight adjusts lighting conditions in images for dramatic or natural effects.
    image, editing, bria, fibo, relighting

    Use cases:
    - Adjust lighting in photos
    - Change illumination conditions
    - Create dramatic lighting effects
    - Relight scenes for better ambiance
    - Transform lighting in images
    """

    class LightType(Enum):
        """
        The quality/style/time of day.
        """
        MIDDAY = "midday"
        BLUE_HOUR_LIGHT = "blue hour light"
        LOW_ANGLE_SUNLIGHT = "low-angle sunlight"
        SUNRISE_LIGHT = "sunrise light"
        SPOTLIGHT_ON_SUBJECT = "spotlight on subject"
        OVERCAST_LIGHT = "overcast light"
        SOFT_OVERCAST_DAYLIGHT_LIGHTING = "soft overcast daylight lighting"
        CLOUD_FILTERED_LIGHTING = "cloud-filtered lighting"
        FOG_DIFFUSED_LIGHTING = "fog-diffused lighting"
        MOONLIGHT_LIGHTING = "moonlight lighting"
        STARLIGHT_NIGHTTIME = "starlight nighttime"
        SOFT_BOKEH_LIGHTING = "soft bokeh lighting"
        HARSH_STUDIO_LIGHTING = "harsh studio lighting"


    light_type: LightType = Field(
        default="", description="The quality/style/time of day."
    )
    light_direction: str = Field(
        default="", description="Where the light comes from."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The source image."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "light_type": self.light_type.value,
            "light_direction": self.light_direction,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/fibo-edit/relight",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class BriaFiboEditReseason(FALNode):
    """
    Bria FIBO Edit Reseason changes the seasonal appearance of outdoor scenes in images.
    image, editing, bria, fibo, seasonal

    Use cases:
    - Change seasons in outdoor photos
    - Transform summer to winter scenes
    - Modify seasonal appearance
    - Create seasonal variations
    - Generate different season versions
    """

    class Season(Enum):
        """
        The desired season.
        """
        SPRING = "spring"
        SUMMER = "summer"
        AUTUMN = "autumn"
        WINTER = "winter"


    season: Season = Field(
        default="", description="The desired season."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The source image."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "season": self.season.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/fibo-edit/reseason",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class BriaFiboEditRewriteText(FALNode):
    """
    Bria FIBO Edit Rewrite Text modifies or replaces text content within images naturally.
    image, editing, bria, fibo, text-editing

    Use cases:
    - Change text in images
    - Replace written content in photos
    - Modify signs and labels
    - Update text naturally in scenes
    - Edit textual elements in images
    """

    new_text: str = Field(
        default="", description="The new text string to appear in the image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The source image."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "new_text": self.new_text,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/fibo-edit/rewrite_text",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class BriaFiboEditSketchToColoredImage(FALNode):
    """
    Bria FIBO Edit Sketch to Colored Image transforms sketches and line art into full-color images.
    image, editing, bria, fibo, sketch-to-image

    Use cases:
    - Convert sketches to colored images
    - Transform line art to full color
    - Generate colored versions of drawings
    - Create realistic images from sketches
    - Produce colored artwork from outlines
    """

    image_url: ImageRef = Field(
        default=ImageRef(), description="The source image."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/fibo-edit/sketch_to_colored_image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class GlmImageImageToImage(FALNode):
    """
    GLM Image image-to-image transforms and modifies images using advanced AI understanding.
    image, transformation, glm, ai-editing

    Use cases:
    - Transform images with GLM AI
    - Apply modifications using advanced understanding
    - Create variations with GLM model
    - Generate modified versions
    - Produce AI-powered transformations
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
    enable_safety_checker: bool = Field(
        default=True, description="Enable NSFW safety checking on the generated images."
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
    image_urls: list[str] = Field(
        default=[], description="URL(s) of the condition image(s) for image-to-image generation. Supports up to 4 URLs for multi-image references."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="If True, the prompt will be enhanced using an LLM for more detailed and higher quality results."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of diffusion denoising steps. More steps generally produce higher quality images."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/glm-image/image-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class GptImage15Edit(FALNode):
    """
    GPT Image 1.5 Edit provides intelligent image editing with GPT-powered understanding and control.
    image, editing, gpt, intelligent, ai-editing

    Use cases:
    - Edit images with GPT intelligence
    - Apply smart modifications to photos
    - Create intelligent edits
    - Transform images with language understanding
    - Produce GPT-powered modifications
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

    class InputFidelity(Enum):
        """
        Input fidelity for the generated image
        """
        LOW = "low"
        HIGH = "high"


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
        default=Quality.HIGH, description="Quality for the generated image"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="Output format for the images"
    )
    input_fidelity: InputFidelity = Field(
        default=InputFidelity.HIGH, description="Input fidelity for the generated image"
    )
    mask_image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the mask image to use for the generation. This indicates what part of the image to edit."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images to use as a reference for the generation."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        mask_image_url_base64 = await context.image_to_base64(self.mask_image_url)
        arguments = {
            "background": self.background.value,
            "num_images": self.num_images,
            "image_size": self.image_size.value,
            "prompt": self.prompt,
            "quality": self.quality.value,
            "output_format": self.output_format.value,
            "input_fidelity": self.input_fidelity.value,
            "mask_image_url": f"data:image/png;base64,{mask_image_url_base64}",
            "sync_mode": self.sync_mode,
            "image_urls": self.image_urls,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/gpt-image-1.5/edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ZImageTurboImageToImage(FALNode):
    """
    Z-Image Turbo image-to-image provides fast image transformations with quality output.
    image, transformation, z-image, turbo, fast

    Use cases:
    - Transform images quickly with Z-Image
    - Apply fast modifications to photos
    - Create rapid image variations
    - Generate speedy transformations
    - Produce quick image modifications
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
        default="auto", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of Image for Image-to-Image generation."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    strength: float = Field(
        default=0.6, description="The strength of the image-to-image conditioning."
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
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "strength": self.strength,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/z-image/turbo/image-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ZImageTurboImageToImageLora(FALNode):
    """
    Z-Image Turbo image-to-image with LoRA enables fast custom-trained model transformations.
    image, transformation, z-image, turbo, lora

    Use cases:
    - Transform images with custom Z-Image models
    - Apply fast specialized modifications
    - Create rapid custom edits
    - Generate quick customized transformations
    - Produce fast fine-tuned modifications
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
        default="auto", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of Image for Image-to-Image generation."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    loras: list[str] = Field(
        default=[], description="List of LoRA weights to apply (maximum 3)."
    )
    strength: float = Field(
        default=0.6, description="The strength of the image-to-image conditioning."
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
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "loras": self.loras,
            "strength": self.strength,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/z-image/turbo/image-to-image/lora",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ZImageTurboInpaint(FALNode):
    """
    Z-Image Turbo Inpaint fills masked regions in images quickly with contextually appropriate content.
    image, inpainting, z-image, turbo, fast

    Use cases:
    - Fill masked regions in images quickly
    - Remove unwanted objects fast
    - Repair image areas with turbo speed
    - Generate quick inpainting results
    - Produce rapid contextual fills
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
    image_size: str = Field(
        default="auto", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    mask_image_url: ImageRef = Field(
        default=ImageRef(), description="URL of Mask for Inpaint generation."
    )
    control_end: float = Field(
        default=0.8, description="The end of the controlnet conditioning."
    )
    control_start: float = Field(
        default=0, description="The start of the controlnet conditioning."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of Image for Inpaint generation."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    strength: float = Field(
        default=1, description="The strength of the inpaint conditioning."
    )
    control_scale: float = Field(
        default=0.75, description="The scale of the controlnet conditioning."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion. Note: this will increase the price by 0.0025 credits per request."
    )
    num_inference_steps: int = Field(
        default=8, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        mask_image_url_base64 = await context.image_to_base64(self.mask_image_url)
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "mask_image_url": f"data:image/png;base64,{mask_image_url_base64}",
            "control_end": self.control_end,
            "control_start": self.control_start,
            "enable_safety_checker": self.enable_safety_checker,
            "num_images": self.num_images,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "strength": self.strength,
            "control_scale": self.control_scale,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/z-image/turbo/inpaint",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ZImageTurboInpaintLora(FALNode):
    """
    Z-Image Turbo Inpaint with LoRA provides fast custom-trained inpainting for specialized tasks.
    image, inpainting, z-image, turbo, lora

    Use cases:
    - Inpaint with custom fast models
    - Fill regions using specialized training
    - Repair images with custom inpainting
    - Generate quick custom fills
    - Produce rapid specialized inpainting
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
    image_size: str = Field(
        default="auto", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    mask_image_url: ImageRef = Field(
        default=ImageRef(), description="URL of Mask for Inpaint generation."
    )
    loras: list[str] = Field(
        default=[], description="List of LoRA weights to apply (maximum 3)."
    )
    control_end: float = Field(
        default=0.8, description="The end of the controlnet conditioning."
    )
    control_start: float = Field(
        default=0, description="The start of the controlnet conditioning."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of Image for Inpaint generation."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    strength: float = Field(
        default=1, description="The strength of the inpaint conditioning."
    )
    control_scale: float = Field(
        default=0.75, description="The scale of the controlnet conditioning."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion. Note: this will increase the price by 0.0025 credits per request."
    )
    num_inference_steps: int = Field(
        default=8, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        mask_image_url_base64 = await context.image_to_base64(self.mask_image_url)
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "mask_image_url": f"data:image/png;base64,{mask_image_url_base64}",
            "loras": self.loras,
            "control_end": self.control_end,
            "control_start": self.control_start,
            "enable_safety_checker": self.enable_safety_checker,
            "num_images": self.num_images,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "strength": self.strength,
            "control_scale": self.control_scale,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/z-image/turbo/inpaint/lora",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ZImageTurboControlnet(FALNode):
    """
    Z-Image Turbo ControlNet provides fast controlled image generation with structural guidance.
    image, controlnet, z-image, turbo, controlled

    Use cases:
    - Generate images with fast structural control
    - Apply quick controlled modifications
    - Create rapid guided generations
    - Transform images with fast ControlNet
    - Produce speedy controlled outputs
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

    class Preprocess(Enum):
        """
        What kind of preprocessing to apply to the image, if any.
        """
        NONE = "none"
        CANNY = "canny"
        DEPTH = "depth"
        POSE = "pose"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    image_size: str = Field(
        default="auto", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    control_end: float = Field(
        default=0.8, description="The end of the controlnet conditioning."
    )
    control_start: float = Field(
        default=0, description="The start of the controlnet conditioning."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of Image for ControlNet generation."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    control_scale: float = Field(
        default=0.75, description="The scale of the controlnet conditioning."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion. Note: this will increase the price by 0.0025 credits per request."
    )
    num_inference_steps: int = Field(
        default=8, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    preprocess: Preprocess = Field(
        default=Preprocess.NONE, description="What kind of preprocessing to apply to the image, if any."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "control_end": self.control_end,
            "control_start": self.control_start,
            "enable_safety_checker": self.enable_safety_checker,
            "num_images": self.num_images,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "control_scale": self.control_scale,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "preprocess": self.preprocess.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/z-image/turbo/controlnet",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ZImageTurboControlnetLora(FALNode):
    """
    Z-Image Turbo ControlNet with LoRA combines fast controlled generation with custom models.
    image, controlnet, z-image, turbo, lora

    Use cases:
    - Generate with fast custom ControlNet
    - Apply quick specialized controlled generation
    - Create rapid custom guided outputs
    - Transform images with fast custom control
    - Produce speedy fine-tuned controlled results
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

    class Preprocess(Enum):
        """
        What kind of preprocessing to apply to the image, if any.
        """
        NONE = "none"
        CANNY = "canny"
        DEPTH = "depth"
        POSE = "pose"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    image_size: str = Field(
        default="auto", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    loras: list[str] = Field(
        default=[], description="List of LoRA weights to apply (maximum 3)."
    )
    control_end: float = Field(
        default=0.8, description="The end of the controlnet conditioning."
    )
    control_start: float = Field(
        default=0, description="The start of the controlnet conditioning."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of Image for ControlNet generation."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    control_scale: float = Field(
        default=0.75, description="The scale of the controlnet conditioning."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion. Note: this will increase the price by 0.0025 credits per request."
    )
    num_inference_steps: int = Field(
        default=8, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    preprocess: Preprocess = Field(
        default=Preprocess.NONE, description="What kind of preprocessing to apply to the image, if any."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "loras": self.loras,
            "control_end": self.control_end,
            "control_start": self.control_start,
            "enable_safety_checker": self.enable_safety_checker,
            "num_images": self.num_images,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "control_scale": self.control_scale,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "preprocess": self.preprocess.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/z-image/turbo/controlnet/lora",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class AiFaceSwapImage(FALNode):
    """
    AI Face Swap replaces faces in images with source faces while maintaining natural appearance.
    image, face-swap, ai, face-manipulation

    Use cases:
    - Swap faces between images
    - Replace faces in photos
    - Create face-swapped variations
    - Generate face replacement results
    - Produce face-substituted images
    """

    enable_occlusion_prevention: bool = Field(
        default=False, description="Enable occlusion prevention for handling faces covered by hands/objects. Warning: Enabling this runs an occlusion-aware model which costs 2x more."
    )
    source_face_url: ImageRef = Field(
        default=ImageRef(), description="Source face image. Allowed items: bmp, jpeg, png, tiff, webp"
    )
    target_image_url: ImageRef = Field(
        default=ImageRef(), description="Target image URL. Allowed items: bmp, jpeg, png, tiff, webp"
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        source_face_url_base64 = await context.image_to_base64(self.source_face_url)
        target_image_url_base64 = await context.image_to_base64(self.target_image_url)
        arguments = {
            "enable_occlusion_prevention": self.enable_occlusion_prevention,
            "source_face_url": f"data:image/png;base64,{source_face_url_base64}",
            "target_image_url": f"data:image/png;base64,{target_image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="half-moon-ai/ai-face-swap/faceswapimage",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["image"]

class AiHomeStyle(FALNode):
    """
    AI Home Style transforms interior spaces with different design styles and aesthetics.
    image, interior-design, style-transfer, home, decoration

    Use cases:
    - Transform interior design styles
    - Apply different home aesthetics
    - Create styled room variations
    - Generate interior design options
    - Produce home styling transformations
    """

    class OutputFormat(Enum):
        """
        The format of the generated image. Choose from: 'jpeg' or 'png'.
        """
        JPEG = "jpeg"
        PNG = "png"

    class Style(Enum):
        """
        Style for furniture and decor
        """
        MINIMALISTIC_INTERIOR = "minimalistic-interior"
        FARMHOUSE_INTERIOR = "farmhouse-interior"
        LUXURY_INTERIOR = "luxury-interior"
        MODERN_INTERIOR = "modern-interior"
        ZEN_INTERIOR = "zen-interior"
        MID_CENTURY_INTERIOR = "mid century-interior"
        AIRBNB_INTERIOR = "airbnb-interior"
        COZY_INTERIOR = "cozy-interior"
        RUSTIC_INTERIOR = "rustic-interior"
        CHRISTMAS_INTERIOR = "christmas-interior"
        BOHEMIAN_INTERIOR = "bohemian-interior"
        TROPICAL_INTERIOR = "tropical-interior"
        INDUSTRIAL_INTERIOR = "industrial-interior"
        JAPANESE_INTERIOR = "japanese-interior"
        VINTAGE_INTERIOR = "vintage-interior"
        LOFT_INTERIOR = "loft-interior"
        HALLOWEEN_INTERIOR = "halloween-interior"
        SOHO_INTERIOR = "soho-interior"
        BAROQUE_INTERIOR = "baroque-interior"
        KIDS_ROOM_INTERIOR = "kids room-interior"
        GIRLS_ROOM_INTERIOR = "girls room-interior"
        BOYS_ROOM_INTERIOR = "boys room-interior"
        SCANDINAVIAN_INTERIOR = "scandinavian-interior"
        FRENCH_COUNTRY_INTERIOR = "french country-interior"
        MEDITERRANEAN_INTERIOR = "mediterranean-interior"
        CYBERPUNK_INTERIOR = "cyberpunk-interior"
        HOT_PINK_INTERIOR = "hot pink-interior"
        BIOPHILIC_INTERIOR = "biophilic-interior"
        ANCIENT_EGYPT_INTERIOR = "ancient egypt-interior"
        PIXEL_INTERIOR = "pixel-interior"
        ART_DECO_INTERIOR = "art deco-interior"
        MODERN_EXTERIOR = "modern-exterior"
        MINIMALISTIC_EXTERIOR = "minimalistic-exterior"
        FARMHOUSE_EXTERIOR = "farmhouse-exterior"
        COZY_EXTERIOR = "cozy-exterior"
        LUXURY_EXTERIOR = "luxury-exterior"
        COLONIAL_EXTERIOR = "colonial-exterior"
        ZEN_EXTERIOR = "zen-exterior"
        ASIAN_EXTERIOR = "asian-exterior"
        CREEPY_EXTERIOR = "creepy-exterior"
        AIRSTONE_EXTERIOR = "airstone-exterior"
        ANCIENT_GREEK_EXTERIOR = "ancient greek-exterior"
        ART_DECO_EXTERIOR = "art deco-exterior"
        BRUTALIST_EXTERIOR = "brutalist-exterior"
        CHRISTMAS_LIGHTS_EXTERIOR = "christmas lights-exterior"
        CONTEMPORARY_EXTERIOR = "contemporary-exterior"
        COTTAGE_EXTERIOR = "cottage-exterior"
        DUTCH_COLONIAL_EXTERIOR = "dutch colonial-exterior"
        FEDERAL_COLONIAL_EXTERIOR = "federal colonial-exterior"
        FIRE_EXTERIOR = "fire-exterior"
        FRENCH_PROVINCIAL_EXTERIOR = "french provincial-exterior"
        FULL_GLASS_EXTERIOR = "full glass-exterior"
        GEORGIAN_COLONIAL_EXTERIOR = "georgian colonial-exterior"
        GOTHIC_EXTERIOR = "gothic-exterior"
        GREEK_REVIVAL_EXTERIOR = "greek revival-exterior"
        ICE_EXTERIOR = "ice-exterior"
        ITALIANATE_EXTERIOR = "italianate-exterior"
        MEDITERRANEAN_EXTERIOR = "mediterranean-exterior"
        MIDCENTURY_EXTERIOR = "midcentury-exterior"
        MIDDLE_EASTERN_EXTERIOR = "middle eastern-exterior"
        MINECRAFT_EXTERIOR = "minecraft-exterior"
        MOROCCO_EXTERIOR = "morocco-exterior"
        NEOCLASSICAL_EXTERIOR = "neoclassical-exterior"
        SPANISH_EXTERIOR = "spanish-exterior"
        TUDOR_EXTERIOR = "tudor-exterior"
        UNDERWATER_EXTERIOR = "underwater-exterior"
        WINTER_EXTERIOR = "winter-exterior"
        YARD_LIGHTING_EXTERIOR = "yard lighting-exterior"

    class ArchitectureType(Enum):
        """
        Type of architecture for appropriate furniture selection
        """
        LIVING_ROOM_INTERIOR = "living room-interior"
        BEDROOM_INTERIOR = "bedroom-interior"
        KITCHEN_INTERIOR = "kitchen-interior"
        DINING_ROOM_INTERIOR = "dining room-interior"
        BATHROOM_INTERIOR = "bathroom-interior"
        LAUNDRY_ROOM_INTERIOR = "laundry room-interior"
        HOME_OFFICE_INTERIOR = "home office-interior"
        STUDY_ROOM_INTERIOR = "study room-interior"
        DORM_ROOM_INTERIOR = "dorm room-interior"
        COFFEE_SHOP_INTERIOR = "coffee shop-interior"
        GAMING_ROOM_INTERIOR = "gaming room-interior"
        RESTAURANT_INTERIOR = "restaurant-interior"
        OFFICE_INTERIOR = "office-interior"
        ATTIC_INTERIOR = "attic-interior"
        TOILET_INTERIOR = "toilet-interior"
        OTHER_INTERIOR = "other-interior"
        HOUSE_EXTERIOR = "house-exterior"
        VILLA_EXTERIOR = "villa-exterior"
        BACKYARD_EXTERIOR = "backyard-exterior"
        COURTYARD_EXTERIOR = "courtyard-exterior"
        RANCH_EXTERIOR = "ranch-exterior"
        OFFICE_EXTERIOR = "office-exterior"
        RETAIL_EXTERIOR = "retail-exterior"
        TOWER_EXTERIOR = "tower-exterior"
        APARTMENT_EXTERIOR = "apartment-exterior"
        SCHOOL_EXTERIOR = "school-exterior"
        MUSEUM_EXTERIOR = "museum-exterior"
        COMMERCIAL_EXTERIOR = "commercial-exterior"
        RESIDENTIAL_EXTERIOR = "residential-exterior"
        OTHER_EXTERIOR = "other-exterior"

    class ColorPalette(Enum):
        """
        Color palette for furniture and decor
        """
        SURPRISE_ME = "surprise me"
        GOLDEN_BEIGE = "golden beige"
        REFINED_BLUES = "refined blues"
        DUSKY_ELEGANCE = "dusky elegance"
        EMERALD_CHARM = "emerald charm"
        CRIMSON_LUXURY = "crimson luxury"
        GOLDEN_SAPPHIRE = "golden sapphire"
        SOFT_PASTURES = "soft pastures"
        CANDY_SKY = "candy sky"
        PEACH_MEADOW = "peach meadow"
        MUTED_SANDS = "muted sands"
        OCEAN_BREEZE = "ocean breeze"
        FROSTED_PASTELS = "frosted pastels"
        SPRING_BLOOM = "spring bloom"
        GENTLE_HORIZON = "gentle horizon"
        SEASIDE_BREEZE = "seaside breeze"
        AZURE_COAST = "azure coast"
        GOLDEN_SHORE = "golden shore"
        MEDITERRANEAN_GEM = "mediterranean gem"
        OCEAN_SERENITY = "ocean serenity"
        SERENE_BLUSH = "serene blush"
        MUTED_HORIZON = "muted horizon"
        PASTEL_SHORES = "pastel shores"
        DUSKY_CALM = "dusky calm"
        WOODLAND_RETREAT = "woodland retreat"
        MEADOW_GLOW = "meadow glow"
        FOREST_CANOPY = "forest canopy"
        RIVERBANK_CALM = "riverbank calm"
        EARTHY_TONES = "earthy tones"
        EARTHY_NEUTRALS = "earthy neutrals"
        ARCTIC_MIST = "arctic mist"
        AQUA_DRIFT = "aqua drift"
        BLUSH_BLOOM = "blush bloom"
        CORAL_HAZE = "coral haze"
        RETRO_RUST = "retro rust"
        AUTUMN_GLOW = "autumn glow"
        RUSTIC_CHARM = "rustic charm"
        VINTAGE_SAGE = "vintage sage"
        FADED_PLUM = "faded plum"
        ELECTRIC_LIME = "electric lime"
        VIOLET_PULSE = "violet pulse"
        NEON_SORBET = "neon sorbet"
        AQUA_GLOW = "aqua glow"
        FLUORESCENT_SUNSET = "fluorescent sunset"
        LAVENDER_BLOOM = "lavender bloom"
        PETAL_FRESH = "petal fresh"
        MEADOW_LIGHT = "meadow light"
        SUNNY_PASTURES = "sunny pastures"
        FROSTED_MAUVE = "frosted mauve"
        SNOWY_HEARTH = "snowy hearth"
        ICY_BLUES = "icy blues"
        WINTER_TWILIGHT = "winter twilight"
        EARTHY_HUES = "earthy hues"
        STONE_BALANCE = "stone balance"
        NEUTRAL_SANDS = "neutral sands"
        SLATE_SHADES = "slate shades"


    input_image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image to do architectural styling"
    )
    input_image_strength: float = Field(
        default=0.85, description="Strength of the input image"
    )
    additional_elements: str = Field(
        default="", description="Additional elements to include in the options above (e.g., plants, lighting)"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image. Choose from: 'jpeg' or 'png'."
    )
    style: Style = Field(
        default="", description="Style for furniture and decor"
    )
    architecture_type: ArchitectureType = Field(
        default="", description="Type of architecture for appropriate furniture selection"
    )
    color_palette: ColorPalette = Field(
        default="", description="Color palette for furniture and decor"
    )
    style_image_url: ImageRef = Field(
        default="", description="URL of the style image, optional. If given, other parameters are ignored"
    )
    custom_prompt: str = Field(
        default="", description="Custom prompt for architectural editing, it overrides above options when used"
    )
    enhanced_rendering: str = Field(
        default=False, description="It gives better rendering quality with more processing time, additional cost is 0.01 USD per image"
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        input_image_url_base64 = await context.image_to_base64(self.input_image_url)
        style_image_url_base64 = await context.image_to_base64(self.style_image_url)
        arguments = {
            "input_image_url": f"data:image/png;base64,{input_image_url_base64}",
            "input_image_strength": self.input_image_strength,
            "additional_elements": self.additional_elements,
            "output_format": self.output_format.value,
            "style": self.style.value,
            "architecture_type": self.architecture_type.value,
            "color_palette": self.color_palette.value,
            "style_image_url": f"data:image/png;base64,{style_image_url_base64}",
            "custom_prompt": self.custom_prompt,
            "enhanced_rendering": self.enhanced_rendering,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="half-moon-ai/ai-home/style",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class AiHomeEdit(FALNode):
    """
    AI Home Edit modifies interior spaces with renovations, furniture changes, and design adjustments.
    image, interior-design, editing, home, renovation

    Use cases:
    - Edit interior spaces
    - Modify room furniture and decor
    - Create renovation visualizations
    - Generate design modification options
    - Produce home editing results
    """

    class EditingType(Enum):
        """
        Type of editing. Structural editing only edits structural elements such as windows, walls etc. Virtual staging edits your furniture. Both do full editing including structural and furniture
        """
        STRUCTURAL_EDITING = "structural editing"
        VIRTUAL_STAGING = "virtual staging"
        BOTH = "both"

    class Style(Enum):
        """
        Style for furniture and decor
        """
        MINIMALISTIC_INTERIOR = "minimalistic-interior"
        FARMHOUSE_INTERIOR = "farmhouse-interior"
        LUXURY_INTERIOR = "luxury-interior"
        MODERN_INTERIOR = "modern-interior"
        ZEN_INTERIOR = "zen-interior"
        MID_CENTURY_INTERIOR = "mid century-interior"
        AIRBNB_INTERIOR = "airbnb-interior"
        COZY_INTERIOR = "cozy-interior"
        RUSTIC_INTERIOR = "rustic-interior"
        CHRISTMAS_INTERIOR = "christmas-interior"
        BOHEMIAN_INTERIOR = "bohemian-interior"
        TROPICAL_INTERIOR = "tropical-interior"
        INDUSTRIAL_INTERIOR = "industrial-interior"
        JAPANESE_INTERIOR = "japanese-interior"
        VINTAGE_INTERIOR = "vintage-interior"
        LOFT_INTERIOR = "loft-interior"
        HALLOWEEN_INTERIOR = "halloween-interior"
        SOHO_INTERIOR = "soho-interior"
        BAROQUE_INTERIOR = "baroque-interior"
        KIDS_ROOM_INTERIOR = "kids room-interior"
        GIRLS_ROOM_INTERIOR = "girls room-interior"
        BOYS_ROOM_INTERIOR = "boys room-interior"
        SCANDINAVIAN_INTERIOR = "scandinavian-interior"
        FRENCH_COUNTRY_INTERIOR = "french country-interior"
        MEDITERRANEAN_INTERIOR = "mediterranean-interior"
        CYBERPUNK_INTERIOR = "cyberpunk-interior"
        HOT_PINK_INTERIOR = "hot pink-interior"
        BIOPHILIC_INTERIOR = "biophilic-interior"
        ANCIENT_EGYPT_INTERIOR = "ancient egypt-interior"
        PIXEL_INTERIOR = "pixel-interior"
        ART_DECO_INTERIOR = "art deco-interior"
        MODERN_EXTERIOR = "modern-exterior"
        MINIMALISTIC_EXTERIOR = "minimalistic-exterior"
        FARMHOUSE_EXTERIOR = "farmhouse-exterior"
        COZY_EXTERIOR = "cozy-exterior"
        LUXURY_EXTERIOR = "luxury-exterior"
        COLONIAL_EXTERIOR = "colonial-exterior"
        ZEN_EXTERIOR = "zen-exterior"
        ASIAN_EXTERIOR = "asian-exterior"
        CREEPY_EXTERIOR = "creepy-exterior"
        AIRSTONE_EXTERIOR = "airstone-exterior"
        ANCIENT_GREEK_EXTERIOR = "ancient greek-exterior"
        ART_DECO_EXTERIOR = "art deco-exterior"
        BRUTALIST_EXTERIOR = "brutalist-exterior"
        CHRISTMAS_LIGHTS_EXTERIOR = "christmas lights-exterior"
        CONTEMPORARY_EXTERIOR = "contemporary-exterior"
        COTTAGE_EXTERIOR = "cottage-exterior"
        DUTCH_COLONIAL_EXTERIOR = "dutch colonial-exterior"
        FEDERAL_COLONIAL_EXTERIOR = "federal colonial-exterior"
        FIRE_EXTERIOR = "fire-exterior"
        FRENCH_PROVINCIAL_EXTERIOR = "french provincial-exterior"
        FULL_GLASS_EXTERIOR = "full glass-exterior"
        GEORGIAN_COLONIAL_EXTERIOR = "georgian colonial-exterior"
        GOTHIC_EXTERIOR = "gothic-exterior"
        GREEK_REVIVAL_EXTERIOR = "greek revival-exterior"
        ICE_EXTERIOR = "ice-exterior"
        ITALIANATE_EXTERIOR = "italianate-exterior"
        MEDITERRANEAN_EXTERIOR = "mediterranean-exterior"
        MIDCENTURY_EXTERIOR = "midcentury-exterior"
        MIDDLE_EASTERN_EXTERIOR = "middle eastern-exterior"
        MINECRAFT_EXTERIOR = "minecraft-exterior"
        MOROCCO_EXTERIOR = "morocco-exterior"
        NEOCLASSICAL_EXTERIOR = "neoclassical-exterior"
        SPANISH_EXTERIOR = "spanish-exterior"
        TUDOR_EXTERIOR = "tudor-exterior"
        UNDERWATER_EXTERIOR = "underwater-exterior"
        WINTER_EXTERIOR = "winter-exterior"
        YARD_LIGHTING_EXTERIOR = "yard lighting-exterior"

    class OutputFormat(Enum):
        """
        The format of the generated image. Choose from: 'jpeg' or 'png'.
        """
        JPEG = "jpeg"
        PNG = "png"

    class ArchitectureType(Enum):
        """
        Type of architecture for appropriate furniture selection
        """
        LIVING_ROOM_INTERIOR = "living room-interior"
        BEDROOM_INTERIOR = "bedroom-interior"
        KITCHEN_INTERIOR = "kitchen-interior"
        DINING_ROOM_INTERIOR = "dining room-interior"
        BATHROOM_INTERIOR = "bathroom-interior"
        LAUNDRY_ROOM_INTERIOR = "laundry room-interior"
        HOME_OFFICE_INTERIOR = "home office-interior"
        STUDY_ROOM_INTERIOR = "study room-interior"
        DORM_ROOM_INTERIOR = "dorm room-interior"
        COFFEE_SHOP_INTERIOR = "coffee shop-interior"
        GAMING_ROOM_INTERIOR = "gaming room-interior"
        RESTAURANT_INTERIOR = "restaurant-interior"
        OFFICE_INTERIOR = "office-interior"
        ATTIC_INTERIOR = "attic-interior"
        TOILET_INTERIOR = "toilet-interior"
        OTHER_INTERIOR = "other-interior"
        HOUSE_EXTERIOR = "house-exterior"
        VILLA_EXTERIOR = "villa-exterior"
        BACKYARD_EXTERIOR = "backyard-exterior"
        COURTYARD_EXTERIOR = "courtyard-exterior"
        RANCH_EXTERIOR = "ranch-exterior"
        OFFICE_EXTERIOR = "office-exterior"
        RETAIL_EXTERIOR = "retail-exterior"
        TOWER_EXTERIOR = "tower-exterior"
        APARTMENT_EXTERIOR = "apartment-exterior"
        SCHOOL_EXTERIOR = "school-exterior"
        MUSEUM_EXTERIOR = "museum-exterior"
        COMMERCIAL_EXTERIOR = "commercial-exterior"
        RESIDENTIAL_EXTERIOR = "residential-exterior"
        OTHER_EXTERIOR = "other-exterior"

    class ColorPalette(Enum):
        """
        Color palette for furniture and decor
        """
        SURPRISE_ME = "surprise me"
        GOLDEN_BEIGE = "golden beige"
        REFINED_BLUES = "refined blues"
        DUSKY_ELEGANCE = "dusky elegance"
        EMERALD_CHARM = "emerald charm"
        CRIMSON_LUXURY = "crimson luxury"
        GOLDEN_SAPPHIRE = "golden sapphire"
        SOFT_PASTURES = "soft pastures"
        CANDY_SKY = "candy sky"
        PEACH_MEADOW = "peach meadow"
        MUTED_SANDS = "muted sands"
        OCEAN_BREEZE = "ocean breeze"
        FROSTED_PASTELS = "frosted pastels"
        SPRING_BLOOM = "spring bloom"
        GENTLE_HORIZON = "gentle horizon"
        SEASIDE_BREEZE = "seaside breeze"
        AZURE_COAST = "azure coast"
        GOLDEN_SHORE = "golden shore"
        MEDITERRANEAN_GEM = "mediterranean gem"
        OCEAN_SERENITY = "ocean serenity"
        SERENE_BLUSH = "serene blush"
        MUTED_HORIZON = "muted horizon"
        PASTEL_SHORES = "pastel shores"
        DUSKY_CALM = "dusky calm"
        WOODLAND_RETREAT = "woodland retreat"
        MEADOW_GLOW = "meadow glow"
        FOREST_CANOPY = "forest canopy"
        RIVERBANK_CALM = "riverbank calm"
        EARTHY_TONES = "earthy tones"
        EARTHY_NEUTRALS = "earthy neutrals"
        ARCTIC_MIST = "arctic mist"
        AQUA_DRIFT = "aqua drift"
        BLUSH_BLOOM = "blush bloom"
        CORAL_HAZE = "coral haze"
        RETRO_RUST = "retro rust"
        AUTUMN_GLOW = "autumn glow"
        RUSTIC_CHARM = "rustic charm"
        VINTAGE_SAGE = "vintage sage"
        FADED_PLUM = "faded plum"
        ELECTRIC_LIME = "electric lime"
        VIOLET_PULSE = "violet pulse"
        NEON_SORBET = "neon sorbet"
        AQUA_GLOW = "aqua glow"
        FLUORESCENT_SUNSET = "fluorescent sunset"
        LAVENDER_BLOOM = "lavender bloom"
        PETAL_FRESH = "petal fresh"
        MEADOW_LIGHT = "meadow light"
        SUNNY_PASTURES = "sunny pastures"
        FROSTED_MAUVE = "frosted mauve"
        SNOWY_HEARTH = "snowy hearth"
        ICY_BLUES = "icy blues"
        WINTER_TWILIGHT = "winter twilight"
        EARTHY_HUES = "earthy hues"
        STONE_BALANCE = "stone balance"
        NEUTRAL_SANDS = "neutral sands"
        SLATE_SHADES = "slate shades"


    input_image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image to do architectural editing"
    )
    editing_type: EditingType = Field(
        default="", description="Type of editing. Structural editing only edits structural elements such as windows, walls etc. Virtual staging edits your furniture. Both do full editing including structural and furniture"
    )
    style: Style = Field(
        default="", description="Style for furniture and decor"
    )
    additional_elements: str = Field(
        default="", description="Additional elements to include in the options above (e.g., plants, lighting)"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image. Choose from: 'jpeg' or 'png'."
    )
    architecture_type: ArchitectureType = Field(
        default="", description="Type of architecture for appropriate furniture selection"
    )
    color_palette: ColorPalette = Field(
        default="", description="Color palette for furniture and decor"
    )
    custom_prompt: str = Field(
        default="", description="Custom prompt for architectural editing, it overrides above options when used"
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        input_image_url_base64 = await context.image_to_base64(self.input_image_url)
        arguments = {
            "input_image_url": f"data:image/png;base64,{input_image_url_base64}",
            "editing_type": self.editing_type.value,
            "style": self.style.value,
            "additional_elements": self.additional_elements,
            "output_format": self.output_format.value,
            "architecture_type": self.architecture_type.value,
            "color_palette": self.color_palette.value,
            "custom_prompt": self.custom_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="half-moon-ai/ai-home/edit",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class AiBabyAndAgingGeneratorSingle(FALNode):
    """
    AI Baby and Aging Generator Single shows age progression or regression for a single person.
    image, aging, age-progression, face-manipulation

    Use cases:
    - Show age progression of person
    - Generate younger or older versions
    - Create aging visualizations
    - Produce age transformation results
    - Visualize person at different ages
    """

    class OutputFormat(Enum):
        """
        The format of the generated image. Choose from: 'jpeg' or 'png'.
        """
        JPEG = "jpeg"
        PNG = "png"

    class AgeGroup(Enum):
        """
        Age group for the generated image. Choose from: 'baby' (0-12 months), 'toddler' (1-3 years), 'preschool' (3-5 years), 'gradeschooler' (6-12 years), 'teen' (13-19 years), 'adult' (20-40 years), 'mid' (40-60 years), 'senior' (60+ years).
        """
        BABY = "baby"
        TODDLER = "toddler"
        PRESCHOOL = "preschool"
        GRADESCHOOLER = "gradeschooler"
        TEEN = "teen"
        ADULT = "adult"
        MID = "mid"
        SENIOR = "senior"

    class Gender(Enum):
        """
        Gender for the generated image. Choose from: 'male' or 'female'.
        """
        MALE = "male"
        FEMALE = "female"


    prompt: str = Field(
        default="a newborn baby, well dressed", description="Text prompt to guide the image generation"
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="", description="The size of the generated image"
    )
    id_image_urls: list[str] = Field(
        default=[], description="List of ID images for single mode (or general reference images)"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image. Choose from: 'jpeg' or 'png'."
    )
    age_group: AgeGroup = Field(
        default="", description="Age group for the generated image. Choose from: 'baby' (0-12 months), 'toddler' (1-3 years), 'preschool' (3-5 years), 'gradeschooler' (6-12 years), 'teen' (13-19 years), 'adult' (20-40 years), 'mid' (40-60 years), 'senior' (60+ years)."
    )
    gender: Gender = Field(
        default="", description="Gender for the generated image. Choose from: 'male' or 'female'."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. If None, a random seed will be used"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "id_image_urls": self.id_image_urls,
            "output_format": self.output_format.value,
            "age_group": self.age_group.value,
            "gender": self.gender.value,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="half-moon-ai/ai-baby-and-aging-generator/single",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image"]

class AiBabyAndAgingGeneratorMulti(FALNode):
    """
    AI Baby and Aging Generator Multi shows age progression or regression for multiple people in one image.
    image, aging, age-progression, multi-face

    Use cases:
    - Show age progression for multiple people
    - Generate family aging visualizations
    - Create multi-person aging results
    - Produce group age transformations
    - Visualize multiple people at different ages
    """

    class OutputFormat(Enum):
        """
        The format of the generated image. Choose from: 'jpeg' or 'png'.
        """
        JPEG = "jpeg"
        PNG = "png"

    class AgeGroup(Enum):
        """
        Age group for the generated image. Choose from: 'baby' (0-12 months), 'toddler' (1-3 years), 'preschool' (3-5 years), 'gradeschooler' (6-12 years), 'teen' (13-19 years), 'adult' (20-40 years), 'mid' (40-60 years), 'senior' (60+ years).
        """
        BABY = "baby"
        TODDLER = "toddler"
        PRESCHOOL = "preschool"
        GRADESCHOOLER = "gradeschooler"
        TEEN = "teen"
        ADULT = "adult"
        MID = "mid"
        SENIOR = "senior"

    class Gender(Enum):
        """
        Gender for the generated image. Choose from: 'male' or 'female'.
        """
        MALE = "male"
        FEMALE = "female"


    prompt: str = Field(
        default="a newborn baby, well dressed", description="Text prompt to guide the image generation"
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="", description="The size of the generated image"
    )
    father_weight: float = Field(
        default=0.5, description="Weight of the father's influence in multi mode generation"
    )
    mother_image_urls: list[str] = Field(
        default=[], description="List of mother images for multi mode"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image. Choose from: 'jpeg' or 'png'."
    )
    age_group: AgeGroup = Field(
        default="", description="Age group for the generated image. Choose from: 'baby' (0-12 months), 'toddler' (1-3 years), 'preschool' (3-5 years), 'gradeschooler' (6-12 years), 'teen' (13-19 years), 'adult' (20-40 years), 'mid' (40-60 years), 'senior' (60+ years)."
    )
    gender: Gender = Field(
        default="", description="Gender for the generated image. Choose from: 'male' or 'female'."
    )
    father_image_urls: list[str] = Field(
        default=[], description="List of father images for multi mode"
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. If None, a random seed will be used"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "father_weight": self.father_weight,
            "mother_image_urls": self.mother_image_urls,
            "output_format": self.output_format.value,
            "age_group": self.age_group.value,
            "gender": self.gender.value,
            "father_image_urls": self.father_image_urls,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="half-moon-ai/ai-baby-and-aging-generator/multi",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image"]

class WanV26ImageToImage(FALNode):
    """
    Wan v2.6 image-to-image provides high-quality image transformations with advanced AI capabilities.
    image, transformation, wan, v2.6, quality

    Use cases:
    - Transform images with Wan v2.6
    - Apply quality modifications to photos
    - Create high-quality variations
    - Generate advanced transformations
    - Produce quality image modifications
    """

    prompt: str = Field(
        default="", description="Text prompt describing the desired image. Supports Chinese and English. Max 2000 characters. Example: 'Generate an image using the style of image 1 and background of image 2'."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate (1-4). Directly affects billing cost."
    )
    image_size: str = Field(
        default="square_hd", description="Output image size. Use presets like 'square_hd', 'landscape_16_9', 'portrait_9_16', or specify exact dimensions with ImageSize(width=1280, height=720). Total pixels must be between 768*768 and 1280*1280."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Enable content moderation for input and output."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility (0-2147483647). Same seed produces more consistent results."
    )
    image_urls: list[str] = Field(
        default=[], description="Reference images for editing (1-3 images required). Order matters: reference as 'image 1', 'image 2', 'image 3' in prompt. Resolution: 384-5000px each dimension. Max size: 10MB each. Formats: JPEG, JPG, PNG (no alpha), BMP, WEBP."
    )
    negative_prompt: str = Field(
        default="", description="Content to avoid in the generated image. Max 500 characters."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Enable LLM prompt optimization. Significantly improves results for simple prompts but adds 3-4 seconds processing time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "enable_prompt_expansion": self.enable_prompt_expansion,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="wan/v2.6/image-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class StepxEdit2(FALNode):
    """
    StepX Edit 2 provides multi-step image editing with progressive refinement and control.
    image, editing, stepx, progressive, refinement

    Use cases:
    - Edit images with progressive steps
    - Apply multi-stage modifications
    - Create refined edits gradually
    - Transform images with step control
    - Produce progressively refined results
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
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    enable_reflection_mode: bool = Field(
        default=True, description="Enable reflection mode. Reviews outputs, corrects unintended changes, and determines when editing is complete."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The image URL to generate an image from. Needs to match the dimensions of the mask."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_scale: float = Field(
        default=6, description="The true CFG scale. Controls how closely the model follows the prompt."
    )
    num_inference_steps: int = Field(
        default=50, description="The number of inference steps to perform. Recommended: 50."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    enable_thinking_mode: bool = Field(
        default=True, description="Enable thinking mode. Uses multimodal language model knowledge to interpret abstract editing instructions."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "seed": self.seed,
            "enable_reflection_mode": self.enable_reflection_mode,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "enable_thinking_mode": self.enable_thinking_mode,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/stepx-edit2",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class LongcatImageEdit(FALNode):
    """
    Longcat Image Edit transforms images with unique AI-powered modifications and creative control.
    image, editing, longcat, creative

    Use cases:
    - Edit images with Longcat AI
    - Apply creative modifications
    - Create unique image variations
    - Transform images creatively
    - Produce artistic modifications
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
        default="", description="The prompt to edit the image with."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to edit."
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
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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
            application="fal-ai/longcat-image/edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class BytedanceSeedreamV45Edit(FALNode):
    """
    ByteDance SeeDream v4.5 Edit provides advanced image editing with cutting-edge AI technology.
    image, editing, bytedance, seedream, v4.5

    Use cases:
    - Edit images with SeeDream v4.5
    - Apply advanced modifications
    - Create high-quality edits
    - Transform images with latest tech
    - Produce cutting-edge modifications
    """

    prompt: str = Field(
        default="", description="The text prompt used to edit the image"
    )
    num_images: int = Field(
        default=1, description="Number of separate model generations to be run with the prompt."
    )
    image_size: str = Field(
        default="", description="The size of the generated image. Width and height must be between 1920 and 4096, or total number of pixels must be between 2560*1440 and 4096*4096."
    )
    max_images: int = Field(
        default=1, description="If set to a number greater than one, enables multi-image generation. The model will potentially return up to `max_images` images every generation, and in total, `num_images` generations will be carried out. In total, the number of images generated will be between `num_images` and `max_images*num_images`. The total number of images (image inputs + image outputs) must not exceed 15"
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
    image_urls: list[str] = Field(
        default=[], description="List of URLs of input images for editing. Presently, up to 10 image inputs are allowed. If over 10 images are sent, only the last 10 will be used."
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
            "image_urls": self.image_urls,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/bytedance/seedream/v4.5/edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ViduQ2ReferenceToImage(FALNode):
    """
    Vidu Q2 Reference-to-Image generates images based on reference images with style and content transfer.
    image, generation, vidu, reference, style-transfer

    Use cases:
    - Generate images from references
    - Transfer style and content
    - Create reference-based variations
    - Transform using reference images
    - Produce style-transferred results
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
    reference_image_urls: list[str] = Field(
        default=[], description="URLs of the reference images to use for consistent subject appearance"
    )
    seed: int = Field(
        default=-1, description="Random seed for generation"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "reference_image_urls": self.reference_image_urls,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/vidu/q2/reference-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class KlingImageO1(FALNode):
    """
    Kling Image O1 provides advanced image generation and transformation with optimized quality.
    image, generation, kling, o1, optimized

    Use cases:
    - Generate images with Kling O1
    - Transform images with optimization
    - Create optimized quality results
    - Produce advanced image generations
    - Generate with balanced quality-speed
    """

    class Resolution(Enum):
        """
        Image generation resolution. 1K: standard, 2K: high-res.
        """
        VALUE_1K = "1K"
        VALUE_2K = "2K"

    class KlingImageO1AspectRatio(Enum):
        """
        Aspect ratio of generated images. 'auto' intelligently determines based on input content.
        """
        AUTO = "auto"
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"
        RATIO_4_3 = "4:3"
        RATIO_3_4 = "3:4"
        RATIO_3_2 = "3:2"
        RATIO_2_3 = "2:3"
        RATIO_21_9 = "21:9"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="Text prompt for image generation. Reference images using @Image1, @Image2, etc. (or @Image if only one image). Max 2500 characters."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate (1-9)."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_1K, description="Image generation resolution. 1K: standard, 2K: high-res."
    )
    aspect_ratio: KlingImageO1AspectRatio = Field(
        default=KlingImageO1AspectRatio.AUTO, description="Aspect ratio of generated images. 'auto' intelligently determines based on input content."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    elements: list[str] = Field(
        default=[], description="Elements (characters/objects) to include in the image. Reference in prompt as @Element1, @Element2, etc. Maximum 10 total (elements + reference images)."
    )
    image_urls: list[str] = Field(
        default=[], description="List of reference images. Reference images in prompt using @Image1, @Image2, etc. (1-indexed). Max 10 images."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "elements": self.elements,
            "image_urls": self.image_urls,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-image/o1",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEdit2509LoraGalleryShirtDesign(FALNode):
    """
    Qwen Image Edit 2509 Lora Gallery
    editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
        default="Put this design on their shirt", description="Describe what design to put on the shirt. The model will apply the design from your input image onto the person's shirt."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the final input image will be used."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    lora_scale: float = Field(
        default=1, description="The scale factor for the LoRA model. Controls the strength of the LoRA effect."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and won't be saved in history."
    )
    guidance_scale: float = Field(
        default=1, description="The CFG (Classifier Free Guidance) scale. Controls how closely the model follows the prompt."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. Same seed with same prompt will produce same result."
    )
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images: first image is the person wearing a shirt, second image is the design/logo to put on the shirt."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    num_inference_steps: int = Field(
        default=6, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "lora_scale": self.lora_scale,
            "output_format": self.output_format.value,
            "enable_safety_checker": self.enable_safety_checker,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-2509-lora-gallery/shirt-design",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEdit2509LoraGalleryRemoveLighting(FALNode):
    """
    Qwen Image Edit 2509 Lora Gallery
    editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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


    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the final input image will be used."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and won't be saved in history."
    )
    guidance_scale: float = Field(
        default=1, description="The CFG (Classifier Free Guidance) scale. Controls how closely the model follows the prompt."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. Same seed with same prompt will produce same result."
    )
    image_urls: list[str] = Field(
        default=[], description="The URL of the image with lighting/shadows to remove."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    num_inference_steps: int = Field(
        default=6, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "enable_safety_checker": self.enable_safety_checker,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-2509-lora-gallery/remove-lighting",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEdit2509LoraGalleryRemoveElement(FALNode):
    """
    Qwen Image Edit 2509 Lora Gallery
    editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
        default="Remove the specified element from the scene", description="Specify what element(s) to remove from the image (objects, people, text, etc.). The model will cleanly remove the element while maintaining consistency of the rest of the image."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the final input image will be used."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    lora_scale: float = Field(
        default=1, description="The scale factor for the LoRA model. Controls the strength of the LoRA effect."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and won't be saved in history."
    )
    guidance_scale: float = Field(
        default=1, description="The CFG (Classifier Free Guidance) scale. Controls how closely the model follows the prompt."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. Same seed with same prompt will produce same result."
    )
    image_urls: list[str] = Field(
        default=[], description="The URL of the image containing elements to remove."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    num_inference_steps: int = Field(
        default=6, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "lora_scale": self.lora_scale,
            "output_format": self.output_format.value,
            "enable_safety_checker": self.enable_safety_checker,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-2509-lora-gallery/remove-element",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEdit2509LoraGalleryLightingRestoration(FALNode):
    """
    Qwen Image Edit 2509 Lora Gallery
    editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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


    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the final input image will be used."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and won't be saved in history."
    )
    guidance_scale: float = Field(
        default=1, description="The CFG (Classifier Free Guidance) scale. Controls how closely the model follows the prompt."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. Same seed with same prompt will produce same result."
    )
    image_urls: list[str] = Field(
        default=[], description="The URL of the image to restore lighting for."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    num_inference_steps: int = Field(
        default=6, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "enable_safety_checker": self.enable_safety_checker,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-2509-lora-gallery/lighting-restoration",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEdit2509LoraGalleryIntegrateProduct(FALNode):
    """
    Qwen Image Edit 2509 Lora Gallery
    editing, transformation, image-to-image, img2img, lora, professional

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
        default="Blend and integrate the product into the background", description="Describe how to blend and integrate the product/element into the background. The model will automatically correct perspective, lighting and shadows for natural integration."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the final input image will be used."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    lora_scale: float = Field(
        default=1, description="The scale factor for the LoRA model. Controls the strength of the LoRA effect."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and won't be saved in history."
    )
    guidance_scale: float = Field(
        default=1, description="The CFG (Classifier Free Guidance) scale. Controls how closely the model follows the prompt."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. Same seed with same prompt will produce same result."
    )
    image_urls: list[str] = Field(
        default=[], description="The URL of the image with product to integrate into background."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    num_inference_steps: int = Field(
        default=6, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "lora_scale": self.lora_scale,
            "output_format": self.output_format.value,
            "enable_safety_checker": self.enable_safety_checker,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-2509-lora-gallery/integrate-product",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEdit2509LoraGalleryGroupPhoto(FALNode):
    """
    Qwen Image Edit 2509 Lora Gallery
    editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
        default="Two people standing next to each other outside with a landscape background", description="Describe the group photo scene, setting, and style. The model will maintain character consistency and add vintage effects like grain, blur, and retro filters."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the final input image will be used."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    lora_scale: float = Field(
        default=1, description="The scale factor for the LoRA model. Controls the strength of the LoRA effect."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and won't be saved in history."
    )
    guidance_scale: float = Field(
        default=1, description="The CFG (Classifier Free Guidance) scale. Controls how closely the model follows the prompt."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. Same seed with same prompt will produce same result."
    )
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images to combine into a group photo. Provide 2 or more individual portrait images."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    num_inference_steps: int = Field(
        default=6, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "lora_scale": self.lora_scale,
            "output_format": self.output_format.value,
            "enable_safety_checker": self.enable_safety_checker,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-2509-lora-gallery/group-photo",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEdit2509LoraGalleryFaceToFullPortrait(FALNode):
    """
    Qwen Image Edit 2509 Lora Gallery
    editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
        default="Photography. A portrait of the person in professional attire with natural lighting", description="Describe the full portrait you want to generate from the face. Include clothing, setting, pose, and style details."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the final input image will be used."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    lora_scale: float = Field(
        default=1, description="The scale factor for the LoRA model. Controls the strength of the LoRA effect."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and won't be saved in history."
    )
    guidance_scale: float = Field(
        default=1, description="The CFG (Classifier Free Guidance) scale. Controls how closely the model follows the prompt."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. Same seed with same prompt will produce same result."
    )
    image_urls: list[str] = Field(
        default=[], description="The URL of the cropped face image. Provide a close-up face photo."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    num_inference_steps: int = Field(
        default=6, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "lora_scale": self.lora_scale,
            "output_format": self.output_format.value,
            "enable_safety_checker": self.enable_safety_checker,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-2509-lora-gallery/face-to-full-portrait",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEdit2509LoraGalleryAddBackground(FALNode):
    """
    Qwen Image Edit 2509 Lora Gallery
    editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
        default="Remove white background and add a realistic scene behind the object", description="Describe the background/scene you want to add behind the object. The model will remove the white background and add the specified environment."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the final input image will be used."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    lora_scale: float = Field(
        default=1, description="The scale factor for the LoRA model. Controls the strength of the LoRA effect."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and won't be saved in history."
    )
    guidance_scale: float = Field(
        default=1, description="The CFG (Classifier Free Guidance) scale. Controls how closely the model follows the prompt."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. Same seed with same prompt will produce same result."
    )
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images to edit. Provide an image with a white or clean background."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    num_inference_steps: int = Field(
        default=6, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "lora_scale": self.lora_scale,
            "output_format": self.output_format.value,
            "enable_safety_checker": self.enable_safety_checker,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-2509-lora-gallery/add-background",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEdit2509LoraGalleryNextScene(FALNode):
    """
    Qwen Image Edit 2509 Lora Gallery
    editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
        default="Next Scene: The camera moves forward revealing more of the scene", description="Describe the camera movement, framing change, or scene transition. Start with 'Next Scene:' for best results. Examples: camera movements (dolly, push-in, pull-back), framing changes (wide to close-up), new elements entering frame."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the final input image will be used."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    lora_scale: float = Field(
        default=1, description="The scale factor for the LoRA model. Controls the strength of the LoRA effect."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and won't be saved in history."
    )
    guidance_scale: float = Field(
        default=1, description="The CFG (Classifier Free Guidance) scale. Controls how closely the model follows the prompt."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. Same seed with same prompt will produce same result."
    )
    image_urls: list[str] = Field(
        default=[], description="The URL of the image to create the next scene from."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    num_inference_steps: int = Field(
        default=6, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "lora_scale": self.lora_scale,
            "output_format": self.output_format.value,
            "enable_safety_checker": self.enable_safety_checker,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-2509-lora-gallery/next-scene",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEdit2509LoraGalleryMultipleAngles(FALNode):
    """
    Qwen Image Edit 2509 Lora Gallery
    editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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


    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the final input image will be used."
    )
    wide_angle_lens: bool = Field(
        default=False, description="Enable wide-angle lens effect"
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    guidance_scale: float = Field(
        default=1, description="The CFG (Classifier Free Guidance) scale. Controls how closely the model follows the prompt."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    image_urls: list[str] = Field(
        default=[], description="The URL of the image to adjust camera angle for."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    vertical_angle: float = Field(
        default=0, description="Adjust vertical camera angle (-1=bird's-eye view/looking down, 0=neutral, 1=worm's-eye view/looking up)"
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    move_forward: float = Field(
        default=0, description="Move camera forward (0=no movement, 10=close-up)"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    rotate_right_left: float = Field(
        default=0, description="Rotate camera left (positive) or right (negative) in degrees. Positive values rotate left, negative values rotate right."
    )
    lora_scale: float = Field(
        default=1.25, description="The scale factor for the LoRA model. Controls the strength of the camera control effect."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and won't be saved in history."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. Same seed with same prompt will produce same result."
    )
    num_inference_steps: int = Field(
        default=6, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "image_size": self.image_size,
            "wide_angle_lens": self.wide_angle_lens,
            "acceleration": self.acceleration.value,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "vertical_angle": self.vertical_angle,
            "num_images": self.num_images,
            "move_forward": self.move_forward,
            "output_format": self.output_format.value,
            "rotate_right_left": self.rotate_right_left,
            "lora_scale": self.lora_scale,
            "sync_mode": self.sync_mode,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-2509-lora-gallery/multiple-angles",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEditPlusLoraGalleryLightingRestoration(FALNode):
    """
    Qwen Image Edit Plus Lora Gallery
    editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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


    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the final input image will be used."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and won't be saved in history."
    )
    guidance_scale: float = Field(
        default=1, description="The CFG (Classifier Free Guidance) scale. Controls how closely the model follows the prompt."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. Same seed with same prompt will produce same result."
    )
    image_urls: list[str] = Field(
        default=[], description="The URL of the image to restore lighting for."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    num_inference_steps: int = Field(
        default=6, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "enable_safety_checker": self.enable_safety_checker,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-plus-lora-gallery/lighting-restoration",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Moondream3PreviewSegment(FALNode):
    """
    Moondream3 Preview [Segment]
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    spatial_references: list[str] = Field(
        default=[], description="Spatial references to guide the segmentation. By feeding in references you can help the segmentation process. Must be either list of Point object with x and y members, or list of arrays containing either 2 floats (x,y) or 4 floats (x1,y1,x2,y2). **NOTE**: You can also use the [**point endpoint**](https://fal.ai/models/fal-ai/moondream3-preview/point) to get points for the objects, and pass them in here."
    )
    settings: str = Field(
        default="", description="Sampling settings for the segmentation model"
    )
    object: str = Field(
        default="", description="Object to be segmented in the image"
    )
    preview: bool = Field(
        default=False, description="Whether to preview the output and return a binary mask of the image"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image to be processed Max width: 7000px, Max height: 7000px, Timeout: 20.0s"
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "spatial_references": self.spatial_references,
            "settings": self.settings,
            "object": self.object,
            "preview": self.preview,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/moondream3-preview/segment",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Flux2LoraGalleryApartmentStaging(FALNode):
    """
    Flux 2 Lora Gallery
    flux, editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
        default="", description="The prompt to generate a furnished room. Use 'furnish this room' for best results."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the input image will be used."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    lora_scale: float = Field(
        default=1, description="The strength of the apartment staging effect."
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
    image_urls: list[str] = Field(
        default=[], description="The URL of the empty room image to furnish."
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
            "image_urls": self.image_urls,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2-lora-gallery/apartment-staging",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ClarityaiCrystalUpscaler(FALNode):
    """
    Crystal Upscaler
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    creativity: float = Field(
        default=0, description="Creativity level for upscaling"
    )
    scale_factor: float = Field(
        default=2, description="Scale factor"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL to the input image"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "creativity": self.creativity,
            "scale_factor": self.scale_factor,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="clarityai/crystal-upscaler",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ChronoEditLora(FALNode):
    """
    Chrono Edit Lora
    editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class Resolution(Enum):
        """
        The resolution of the output image.
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"

    class OutputFormat(Enum):
        """
        The format of the output image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to edit the image."
    )
    loras: list[str] = Field(
        default=[], description="Optional additional LoRAs to merge for this request (max 3)."
    )
    turbo_mode: bool = Field(
        default=True, description="Enable turbo mode to use for faster inference."
    )
    enable_temporal_reasoning: bool = Field(
        default=False, description="Whether to enable temporal reasoning."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    guidance_scale: float = Field(
        default=1, description="The guidance scale for the inference."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_480P, description="The resolution of the output image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the output image."
    )
    num_temporal_reasoning_steps: int = Field(
        default=8, description="The number of temporal reasoning steps to perform."
    )
    sync_mode: bool = Field(
        default=False, description="Whether to return the image in sync mode."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The image to edit."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt expansion."
    )
    num_inference_steps: int = Field(
        default=8, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The seed for the inference."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "loras": self.loras,
            "turbo_mode": self.turbo_mode,
            "enable_temporal_reasoning": self.enable_temporal_reasoning,
            "enable_safety_checker": self.enable_safety_checker,
            "guidance_scale": self.guidance_scale,
            "resolution": self.resolution.value,
            "output_format": self.output_format.value,
            "num_temporal_reasoning_steps": self.num_temporal_reasoning_steps,
            "sync_mode": self.sync_mode,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/chrono-edit-lora",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ChronoEditLoraGalleryPaintbrush(FALNode):
    """
    Chrono Edit Lora Gallery
    editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class Resolution(Enum):
        """
        The resolution of the output image.
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"

    class OutputFormat(Enum):
        """
        The format of the output image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="Describe how to transform the sketched regions."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_480P, description="The resolution of the output image."
    )
    lora_scale: float = Field(
        default=1, description="The scale factor for the LoRA adapter."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The image to edit."
    )
    sync_mode: bool = Field(
        default=False, description="Whether to return the image in sync mode."
    )
    turbo_mode: bool = Field(
        default=True, description="Enable turbo mode to use faster inference."
    )
    loras: list[str] = Field(
        default=[], description="Optional additional LoRAs to merge (max 3)."
    )
    guidance_scale: float = Field(
        default=1, description="Classifier-free guidance scale."
    )
    num_inference_steps: int = Field(
        default=8, description="Number of denoising steps to run."
    )
    mask_url: ImageRef = Field(
        default=ImageRef(), description="Optional mask image where black areas indicate regions to sketch/paint."
    )
    seed: int = Field(
        default=-1, description="The seed for the inference."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        mask_url_base64 = await context.image_to_base64(self.mask_url)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "lora_scale": self.lora_scale,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "turbo_mode": self.turbo_mode,
            "loras": self.loras,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "mask_url": f"data:image/png;base64,{mask_url_base64}",
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/chrono-edit-lora-gallery/paintbrush",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ChronoEditLoraGalleryUpscaler(FALNode):
    """
    Chrono Edit Lora Gallery
    editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class OutputFormat(Enum):
        """
        The format of the output image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    lora_scale: float = Field(
        default=1, description="The scale factor for the LoRA adapter."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the output image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The image to upscale."
    )
    sync_mode: bool = Field(
        default=False, description="Whether to return the image in sync mode."
    )
    loras: list[str] = Field(
        default=[], description="Optional additional LoRAs to merge (max 3)."
    )
    upscale_factor: float = Field(
        default=2, description="Target scale factor for the output resolution."
    )
    guidance_scale: float = Field(
        default=1, description="The guidance scale for the inference."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of inference steps for the upscaling pass."
    )
    seed: int = Field(
        default=-1, description="The seed for the inference."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "lora_scale": self.lora_scale,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "loras": self.loras,
            "upscale_factor": self.upscale_factor,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/chrono-edit-lora-gallery/upscaler",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Sam3ImageRle(FALNode):
    """
    Sam 3
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="wheel", description="Text prompt for segmentation"
    )
    include_boxes: bool = Field(
        default=False, description="Whether to include bounding boxes for each mask (when available)."
    )
    return_multiple_masks: bool = Field(
        default=False, description="If True, upload and return multiple generated masks as defined by `max_masks`."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image to be segmented"
    )
    sync_mode: bool = Field(
        default=False, description="If True, the media will be returned as a data URI."
    )
    point_prompts: list[str] = Field(
        default=[], description="List of point prompts"
    )
    include_scores: bool = Field(
        default=False, description="Whether to include mask confidence scores."
    )
    max_masks: int = Field(
        default=3, description="Maximum number of masks to return when `return_multiple_masks` is enabled."
    )
    box_prompts: list[str] = Field(
        default=[], description="Box prompt coordinates (x_min, y_min, x_max, y_max). Multiple boxes supported - use object_id to group boxes for the same object or leave empty for separate objects."
    )
    apply_mask: bool = Field(
        default=True, description="Apply the mask on the image."
    )
    text_prompt: str = Field(
        default="", description="[DEPRECATED] Use 'prompt' instead. Kept for backward compatibility."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "include_boxes": self.include_boxes,
            "return_multiple_masks": self.return_multiple_masks,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "point_prompts": self.point_prompts,
            "include_scores": self.include_scores,
            "max_masks": self.max_masks,
            "box_prompts": self.box_prompts,
            "apply_mask": self.apply_mask,
            "text_prompt": self.text_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sam-3/image-rle",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Sam3Image(FALNode):
    """
    Segment Anything Model 3
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="wheel", description="Text prompt for segmentation"
    )
    include_boxes: bool = Field(
        default=False, description="Whether to include bounding boxes for each mask (when available)."
    )
    return_multiple_masks: bool = Field(
        default=False, description="If True, upload and return multiple generated masks as defined by `max_masks`."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image to be segmented"
    )
    sync_mode: bool = Field(
        default=False, description="If True, the media will be returned as a data URI."
    )
    point_prompts: list[str] = Field(
        default=[], description="List of point prompts"
    )
    include_scores: bool = Field(
        default=False, description="Whether to include mask confidence scores."
    )
    max_masks: int = Field(
        default=3, description="Maximum number of masks to return when `return_multiple_masks` is enabled."
    )
    box_prompts: list[str] = Field(
        default=[], description="Box prompt coordinates (x_min, y_min, x_max, y_max). Multiple boxes supported - use object_id to group boxes for the same object or leave empty for separate objects."
    )
    apply_mask: bool = Field(
        default=True, description="Apply the mask on the image."
    )
    text_prompt: str = Field(
        default="", description="[DEPRECATED] Use 'prompt' instead. Kept for backward compatibility."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "include_boxes": self.include_boxes,
            "return_multiple_masks": self.return_multiple_masks,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "point_prompts": self.point_prompts,
            "include_scores": self.include_scores,
            "max_masks": self.max_masks,
            "box_prompts": self.box_prompts,
            "apply_mask": self.apply_mask,
            "text_prompt": self.text_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sam-3/image",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Gemini3ProImagePreviewEdit(FALNode):
    """
    Gemini 3 Pro Image Preview
    editing, transformation, image-to-image, img2img, professional

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
        default="", description="The prompt for image editing."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    enable_web_search: bool = Field(
        default=False, description="Enable web search for the image generation task. This will allow the model to use the latest information from the web to generate the image."
    )
    aspect_ratio: str = Field(
        default="auto", description="The aspect ratio of the generated image."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_1K, description="The resolution of the image to generate."
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
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images to use for image-to-image generation or image editing."
    )
    limit_generations: bool = Field(
        default=False, description="Experimental parameter to limit the number of generations from each round of prompting to 1. Set to `True` to to disregard any instructions in the prompt regarding the number of images to generate."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "enable_web_search": self.enable_web_search,
            "aspect_ratio": self.aspect_ratio,
            "resolution": self.resolution.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "safety_tolerance": self.safety_tolerance.value,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "limit_generations": self.limit_generations,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/gemini-3-pro-image-preview/edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class NanoBananaProEdit(FALNode):
    """
    Nano Banana Pro
    editing, transformation, image-to-image, img2img, professional

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
        default="", description="The prompt for image editing."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    enable_web_search: bool = Field(
        default=False, description="Enable web search for the image generation task. This will allow the model to use the latest information from the web to generate the image."
    )
    aspect_ratio: str = Field(
        default="auto", description="The aspect ratio of the generated image."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_1K, description="The resolution of the image to generate."
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
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images to use for image-to-image generation or image editing."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "enable_web_search": self.enable_web_search,
            "aspect_ratio": self.aspect_ratio,
            "resolution": self.resolution.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "safety_tolerance": self.safety_tolerance.value,
            "seed": self.seed,
            "limit_generations": self.limit_generations,
            "image_urls": self.image_urls,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/nano-banana-pro/edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEditPlusLoraGalleryMultipleAngles(FALNode):
    """
    Qwen Image Edit Plus Lora Gallery
    editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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


    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the final input image will be used."
    )
    wide_angle_lens: bool = Field(
        default=False, description="Enable wide-angle lens effect"
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    guidance_scale: float = Field(
        default=1, description="The CFG (Classifier Free Guidance) scale. Controls how closely the model follows the prompt."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    image_urls: list[str] = Field(
        default=[], description="The URL of the image to adjust camera angle for."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    vertical_angle: float = Field(
        default=0, description="Adjust vertical camera angle (-1=bird's-eye view/looking down, 0=neutral, 1=worm's-eye view/looking up)"
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    move_forward: float = Field(
        default=0, description="Move camera forward (0=no movement, 10=close-up)"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    rotate_right_left: float = Field(
        default=0, description="Rotate camera left (positive) or right (negative) in degrees. Positive values rotate left, negative values rotate right."
    )
    lora_scale: float = Field(
        default=1.25, description="The scale factor for the LoRA model. Controls the strength of the camera control effect."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and won't be saved in history."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. Same seed with same prompt will produce same result."
    )
    num_inference_steps: int = Field(
        default=6, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "image_size": self.image_size,
            "wide_angle_lens": self.wide_angle_lens,
            "acceleration": self.acceleration.value,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "vertical_angle": self.vertical_angle,
            "num_images": self.num_images,
            "move_forward": self.move_forward,
            "output_format": self.output_format.value,
            "rotate_right_left": self.rotate_right_left,
            "lora_scale": self.lora_scale,
            "sync_mode": self.sync_mode,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-plus-lora-gallery/multiple-angles",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEditPlusLoraGalleryShirtDesign(FALNode):
    """
    Qwen Image Edit Plus Lora Gallery
    editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
        default="Put this design on their shirt", description="Describe what design to put on the shirt. The model will apply the design from your input image onto the person's shirt."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the final input image will be used."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    lora_scale: float = Field(
        default=1, description="The scale factor for the LoRA model. Controls the strength of the LoRA effect."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and won't be saved in history."
    )
    guidance_scale: float = Field(
        default=1, description="The CFG (Classifier Free Guidance) scale. Controls how closely the model follows the prompt."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. Same seed with same prompt will produce same result."
    )
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images: first image is the person wearing a shirt, second image is the design/logo to put on the shirt."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    num_inference_steps: int = Field(
        default=6, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "lora_scale": self.lora_scale,
            "output_format": self.output_format.value,
            "enable_safety_checker": self.enable_safety_checker,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-plus-lora-gallery/shirt-design",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEditPlusLoraGalleryRemoveLighting(FALNode):
    """
    Qwen Image Edit Plus Lora Gallery
    editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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


    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the final input image will be used."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and won't be saved in history."
    )
    guidance_scale: float = Field(
        default=1, description="The CFG (Classifier Free Guidance) scale. Controls how closely the model follows the prompt."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. Same seed with same prompt will produce same result."
    )
    image_urls: list[str] = Field(
        default=[], description="The URL of the image with lighting/shadows to remove."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    num_inference_steps: int = Field(
        default=6, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "enable_safety_checker": self.enable_safety_checker,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-plus-lora-gallery/remove-lighting",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEditPlusLoraGalleryRemoveElement(FALNode):
    """
    Qwen Image Edit Plus Lora Gallery
    editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
        default="Remove the specified element from the scene", description="Specify what element(s) to remove from the image (objects, people, text, etc.). The model will cleanly remove the element while maintaining consistency of the rest of the image."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the final input image will be used."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    lora_scale: float = Field(
        default=1, description="The scale factor for the LoRA model. Controls the strength of the LoRA effect."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and won't be saved in history."
    )
    guidance_scale: float = Field(
        default=1, description="The CFG (Classifier Free Guidance) scale. Controls how closely the model follows the prompt."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. Same seed with same prompt will produce same result."
    )
    image_urls: list[str] = Field(
        default=[], description="The URL of the image containing elements to remove."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    num_inference_steps: int = Field(
        default=6, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "lora_scale": self.lora_scale,
            "output_format": self.output_format.value,
            "enable_safety_checker": self.enable_safety_checker,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-plus-lora-gallery/remove-element",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEditPlusLoraGalleryNextScene(FALNode):
    """
    Qwen Image Edit Plus Lora Gallery
    editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
        default="Next Scene: The camera moves forward revealing more of the scene", description="Describe the camera movement, framing change, or scene transition. Start with 'Next Scene:' for best results. Examples: camera movements (dolly, push-in, pull-back), framing changes (wide to close-up), new elements entering frame."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the final input image will be used."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    lora_scale: float = Field(
        default=1, description="The scale factor for the LoRA model. Controls the strength of the LoRA effect."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and won't be saved in history."
    )
    guidance_scale: float = Field(
        default=1, description="The CFG (Classifier Free Guidance) scale. Controls how closely the model follows the prompt."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. Same seed with same prompt will produce same result."
    )
    image_urls: list[str] = Field(
        default=[], description="The URL of the image to create the next scene from."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    num_inference_steps: int = Field(
        default=6, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "lora_scale": self.lora_scale,
            "output_format": self.output_format.value,
            "enable_safety_checker": self.enable_safety_checker,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-plus-lora-gallery/next-scene",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEditPlusLoraGalleryIntegrateProduct(FALNode):
    """
    Qwen Image Edit Plus Lora Gallery
    editing, transformation, image-to-image, img2img, lora, professional

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
        default="Blend and integrate the product into the background", description="Describe how to blend and integrate the product/element into the background. The model will automatically correct perspective, lighting and shadows for natural integration."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the final input image will be used."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    lora_scale: float = Field(
        default=1, description="The scale factor for the LoRA model. Controls the strength of the LoRA effect."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and won't be saved in history."
    )
    guidance_scale: float = Field(
        default=1, description="The CFG (Classifier Free Guidance) scale. Controls how closely the model follows the prompt."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. Same seed with same prompt will produce same result."
    )
    image_urls: list[str] = Field(
        default=[], description="The URL of the image with product to integrate into background."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    num_inference_steps: int = Field(
        default=6, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "lora_scale": self.lora_scale,
            "output_format": self.output_format.value,
            "enable_safety_checker": self.enable_safety_checker,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-plus-lora-gallery/integrate-product",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEditPlusLoraGalleryGroupPhoto(FALNode):
    """
    Qwen Image Edit Plus Lora Gallery
    editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
        default="Two people standing next to each other outside with a landscape background", description="Describe the group photo scene, setting, and style. The model will maintain character consistency and add vintage effects like grain, blur, and retro filters."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the final input image will be used."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    lora_scale: float = Field(
        default=1, description="The scale factor for the LoRA model. Controls the strength of the LoRA effect."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and won't be saved in history."
    )
    guidance_scale: float = Field(
        default=1, description="The CFG (Classifier Free Guidance) scale. Controls how closely the model follows the prompt."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. Same seed with same prompt will produce same result."
    )
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images to combine into a group photo. Provide 2 or more individual portrait images."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    num_inference_steps: int = Field(
        default=6, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "lora_scale": self.lora_scale,
            "output_format": self.output_format.value,
            "enable_safety_checker": self.enable_safety_checker,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-plus-lora-gallery/group-photo",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEditPlusLoraGalleryFaceToFullPortrait(FALNode):
    """
    Qwen Image Edit Plus Lora Gallery
    editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
        default="Photography. A portrait of the person in professional attire with natural lighting", description="Describe the full portrait you want to generate from the face. Include clothing, setting, pose, and style details."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the final input image will be used."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    lora_scale: float = Field(
        default=1, description="The scale factor for the LoRA model. Controls the strength of the LoRA effect."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and won't be saved in history."
    )
    guidance_scale: float = Field(
        default=1, description="The CFG (Classifier Free Guidance) scale. Controls how closely the model follows the prompt."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. Same seed with same prompt will produce same result."
    )
    image_urls: list[str] = Field(
        default=[], description="The URL of the cropped face image. Provide a close-up face photo."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    num_inference_steps: int = Field(
        default=6, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "lora_scale": self.lora_scale,
            "output_format": self.output_format.value,
            "enable_safety_checker": self.enable_safety_checker,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-plus-lora-gallery/face-to-full-portrait",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEditPlusLoraGalleryAddBackground(FALNode):
    """
    Qwen Image Edit Plus Lora Gallery
    editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
        default="Remove white background and add a realistic scene behind the object", description="Describe the background/scene you want to add behind the object. The model will remove the white background and add the specified environment."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the final input image will be used."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    lora_scale: float = Field(
        default=1, description="The scale factor for the LoRA model. Controls the strength of the LoRA effect."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and won't be saved in history."
    )
    guidance_scale: float = Field(
        default=1, description="The CFG (Classifier Free Guidance) scale. Controls how closely the model follows the prompt."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. Same seed with same prompt will produce same result."
    )
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images to edit. Provide an image with a white or clean background."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    num_inference_steps: int = Field(
        default=6, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "lora_scale": self.lora_scale,
            "output_format": self.output_format.value,
            "enable_safety_checker": self.enable_safety_checker,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-plus-lora-gallery/add-background",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ReveFastRemix(FALNode):
    """
    Reve
    editing, transformation, image-to-image, img2img, fast

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class AspectRatio(Enum):
        """
        The desired aspect ratio of the generated image. If not provided, will be smartly chosen by the model.
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
        default="", description="The text description of the desired image. May include XML img tags like <img>0</img> to refer to specific images by their index in the image_urls list."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    aspect_ratio: AspectRatio | None = Field(
        default=None, description="The desired aspect ratio of the generated image. If not provided, will be smartly chosen by the model."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="Output format for the generated image."
    )
    image_urls: list[str] = Field(
        default=[], description="List of URLs of reference images. Must provide between 1 and 6 images (inclusive). Each image must be less than 10 MB. Supports PNG, JPEG, WebP, AVIF, and HEIF formats."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "aspect_ratio": self.aspect_ratio.value if self.aspect_ratio else None,
            "sync_mode": self.sync_mode,
            "output_format": self.output_format.value,
            "image_urls": self.image_urls,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/reve/fast/remix",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ReveFastEdit(FALNode):
    """
    Reve
    editing, transformation, image-to-image, img2img, fast

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class OutputFormat(Enum):
        """
        Output format for the generated image.
        """
        PNG = "png"
        JPEG = "jpeg"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The text description of how to edit the provided image."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="Output format for the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the reference image to edit. Must be publicly accessible or base64 data URI. Supports PNG, JPEG, WebP, AVIF, and HEIF formats."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "sync_mode": self.sync_mode,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/reve/fast/edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ImageAppsV2Outpaint(FALNode):
    """
    Image Outpaint
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class OutputFormat(Enum):
        """
        The format of the output image.
        """
        PNG = "png"
        JPEG = "jpeg"
        JPG = "jpg"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="Optional prompt to guide the outpainting. If provided, it will be appended to the base outpaint instruction. Example: 'with a beautiful sunset in the background'"
    )
    expand_right: int = Field(
        default=0, description="Number of pixels to add as black margin on the right side (0-700)."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate."
    )
    zoom_out_percentage: float = Field(
        default=20, description="Percentage to zoom out the image. If set, the image will be scaled down by this percentage and black margins will be added to maintain original size. Example: 50 means the image will be 50% of original size with black margins filling the rest."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="Image URL to outpaint"
    )
    sync_mode: bool = Field(
        default=False, description="If True, the function will wait for the image to be generated and uploaded before returning the response. If False, the function will return immediately and the image will be generated asynchronously."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    expand_left: int = Field(
        default=0, description="Number of pixels to add as black margin on the left side (0-700)."
    )
    expand_bottom: int = Field(
        default=400, description="Number of pixels to add as black margin on the bottom side (0-700)."
    )
    expand_top: int = Field(
        default=0, description="Number of pixels to add as black margin on the top side (0-700)."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "expand_right": self.expand_right,
            "num_images": self.num_images,
            "zoom_out_percentage": self.zoom_out_percentage,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "expand_left": self.expand_left,
            "expand_bottom": self.expand_bottom,
            "expand_top": self.expand_top,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/image-apps-v2/outpaint",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class FluxVisionUpscaler(FALNode):
    """
    Flux Vision Upscaler
    flux, editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    guidance: float = Field(
        default=1, description="CFG/guidance scale (1-4). Controls how closely the model follows the prompt."
    )
    creativity: float = Field(
        default=0.3, description="The creativity of the model. The higher the creativity, the more the model will deviate from the original. Refers to the denoise strength of the sampling."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to upscale."
    )
    upscale_factor: float = Field(
        default=2, description="The upscale factor (1-4x)."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    seed: str = Field(
        default="", description="The seed to use for the upscale. If not provided, a random seed will be used."
    )
    steps: int = Field(
        default=20, description="Number of inference steps (4-50)."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "guidance": self.guidance,
            "creativity": self.creativity,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "upscale_factor": self.upscale_factor,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "steps": self.steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-vision-upscaler",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Emu3_5ImageEditImage(FALNode):
    """
    Emu 3.5 Image
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
        AUTO = "auto"
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
        default="", description="The prompt to edit the image."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the output image."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.AUTO, description="The aspect ratio of the output image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The image to edit."
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
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/emu-3.5-image/edit-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ChronoEdit(FALNode):
    """
    Chrono Edit
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class Resolution(Enum):
        """
        The resolution of the output image.
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"

    class OutputFormat(Enum):
        """
        The format of the output image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to edit the image."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_480P, description="The resolution of the output image."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the output image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The image to edit."
    )
    turbo_mode: bool = Field(
        default=True, description="Enable turbo mode to use for faster inference."
    )
    num_temporal_reasoning_steps: int = Field(
        default=8, description="The number of temporal reasoning steps to perform."
    )
    sync_mode: bool = Field(
        default=False, description="Whether to return the image in sync mode."
    )
    guidance_scale: float = Field(
        default=1, description="The guidance scale for the inference."
    )
    num_inference_steps: int = Field(
        default=8, description="The number of inference steps to perform."
    )
    enable_temporal_reasoning: bool = Field(
        default=False, description="Whether to enable temporal reasoning."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt expansion."
    )
    seed: int = Field(
        default=-1, description="The seed for the inference."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "turbo_mode": self.turbo_mode,
            "num_temporal_reasoning_steps": self.num_temporal_reasoning_steps,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "enable_temporal_reasoning": self.enable_temporal_reasoning,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/chrono-edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class GptImage1MiniEdit(FALNode):
    """
    GPT Image 1 Mini
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images to use as a reference for the generation."
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
            "image_urls": self.image_urls,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/gpt-image-1-mini/edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ReveRemix(FALNode):
    """
    Reve
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class AspectRatio(Enum):
        """
        The desired aspect ratio of the generated image. If not provided, will be smartly chosen by the model.
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
        default="", description="The text description of the desired image. May include XML img tags like <img>0</img> to refer to specific images by their index in the image_urls list."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    aspect_ratio: AspectRatio | None = Field(
        default=None, description="The desired aspect ratio of the generated image. If not provided, will be smartly chosen by the model."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="Output format for the generated image."
    )
    image_urls: list[str] = Field(
        default=[], description="List of URLs of reference images. Must provide between 1 and 6 images (inclusive). Each image must be less than 10 MB. Supports PNG, JPEG, WebP, AVIF, and HEIF formats."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "aspect_ratio": self.aspect_ratio.value if self.aspect_ratio else None,
            "sync_mode": self.sync_mode,
            "output_format": self.output_format.value,
            "image_urls": self.image_urls,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/reve/remix",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ReveEdit(FALNode):
    """
    Reve
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class OutputFormat(Enum):
        """
        Output format for the generated image.
        """
        PNG = "png"
        JPEG = "jpeg"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The text description of how to edit the provided image."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="Output format for the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the reference image to edit. Must be publicly accessible or base64 data URI. Supports PNG, JPEG, WebP, AVIF, and HEIF formats."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "sync_mode": self.sync_mode,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/reve/edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Image2Pixel(FALNode):
    """
    Image2Pixel
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class DetectMethod(Enum):
        """
        Scale detection method to use.
        """
        AUTO = "auto"
        RUNS = "runs"
        EDGE = "edge"

    class DownscaleMethod(Enum):
        """
        Downscaling method to produce the pixel-art output.
        """
        DOMINANT = "dominant"
        MEDIAN = "median"
        MODE = "mode"
        MEAN = "mean"
        CONTENT_ADAPTIVE = "content-adaptive"

    class BackgroundMode(Enum):
        """
        Controls where to flood-fill from when removing the background.
        """
        EDGES = "edges"
        CORNERS = "corners"
        MIDPOINTS = "midpoints"


    cleanup_morph: bool = Field(
        default=False, description="Apply morphological operations to remove noise."
    )
    auto_color_detect: bool = Field(
        default=False, description="Enable automatic detection of optimal number of colors."
    )
    alpha_threshold: int = Field(
        default=128, description="Alpha binarization threshold (0-255)."
    )
    snap_grid: bool = Field(
        default=True, description="Align output to the pixel grid."
    )
    fixed_palette: list[str] = Field(
        default=[], description="Optional fixed color palette as hex strings (e.g., ['#000000', '#ffffff'])."
    )
    scale: int = Field(
        default=0, description="Force a specific pixel scale. If None, auto-detect."
    )
    cleanup_jaggy: bool = Field(
        default=False, description="Remove isolated diagonal pixels (jaggy edge cleanup)."
    )
    trim_borders: bool = Field(
        default=False, description="Trim borders of the image."
    )
    background_tolerance: int = Field(
        default=0, description="Background tolerance (0-255)."
    )
    detect_method: DetectMethod = Field(
        default=DetectMethod.AUTO, description="Scale detection method to use."
    )
    transparent_background: bool = Field(
        default=False, description="Remove background of the image. This will check for contiguous color regions from the edges after correction and make them transparent."
    )
    downscale_method: DownscaleMethod = Field(
        default=DownscaleMethod.DOMINANT, description="Downscaling method to produce the pixel-art output."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The image URL to process into improved pixel art"
    )
    background_mode: BackgroundMode = Field(
        default=BackgroundMode.CORNERS, description="Controls where to flood-fill from when removing the background."
    )
    max_colors: int = Field(
        default=32, description="Maximum number of colors in the output palette. Set None to disable limit."
    )
    dominant_color_threshold: float = Field(
        default=0.05, description="Dominant color threshold (0.0-1.0)."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "cleanup_morph": self.cleanup_morph,
            "auto_color_detect": self.auto_color_detect,
            "alpha_threshold": self.alpha_threshold,
            "snap_grid": self.snap_grid,
            "fixed_palette": self.fixed_palette,
            "scale": self.scale,
            "cleanup_jaggy": self.cleanup_jaggy,
            "trim_borders": self.trim_borders,
            "background_tolerance": self.background_tolerance,
            "detect_method": self.detect_method.value,
            "transparent_background": self.transparent_background,
            "downscale_method": self.downscale_method.value,
            "sync_mode": self.sync_mode,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "background_mode": self.background_mode.value,
            "max_colors": self.max_colors,
            "dominant_color_threshold": self.dominant_color_threshold,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/image2pixel",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Dreamomni2Edit(FALNode):
    """
    DreamOmni2
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    prompt: str = Field(
        default="", description="The prompt to edit the image."
    )
    image_urls: list[str] = Field(
        default=[], description="List of URLs of input images for editing."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_urls": self.image_urls,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/dreamomni2/edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEditPlusLora(FALNode):
    """
    Qwen Image Edit Plus Lora
    editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class Acceleration(Enum):
        """
        Acceleration level for image generation. Options: 'none', 'regular'. Higher acceleration increases speed. 'regular' balances speed and quality.
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
        default="", description="The prompt to generate the image with"
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="", description="The size of the generated image. If not provided, the size of the final input image will be used to calculate the size of the output image."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. Options: 'none', 'regular'. Higher acceleration increases speed. 'regular' balances speed and quality."
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
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images to edit."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    guidance_scale: float = Field(
        default=4, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "seed": self.seed,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "loras": self.loras,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "guidance_scale": self.guidance_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-plus-lora",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Lucidflux(FALNode):
    """
    Lucidflux
    flux, editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    prompt: str = Field(
        default="", description="The prompt to edit the image."
    )
    guidance: float = Field(
        default=4, description="The guidance to use for the diffusion process."
    )
    target_height: int = Field(
        default=1024, description="The height of the output image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to edit."
    )
    target_width: int = Field(
        default=1024, description="The width of the output image."
    )
    num_inference_steps: int = Field(
        default=50, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    seed: int = Field(
        default=42, description="Seed used for random number generation"
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "guidance": self.guidance,
            "target_height": self.target_height,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "target_width": self.target_width,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/lucidflux",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEditImageToImage(FALNode):
    """
    Qwen Image Edit
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class Acceleration(Enum):
        """
        Acceleration level for image generation. Options: 'none', 'regular'. Higher acceleration increases speed. 'regular' balances speed and quality.
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
        default="", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. Options: 'none', 'regular'. Higher acceleration increases speed. 'regular' balances speed and quality."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to edit."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    strength: float = Field(
        default=0.94, description="Strength of the image-to-image transformation. Lower values preserve more of the original image."
    )
    guidance_scale: float = Field(
        default=4, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=30, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "strength": self.strength,
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
            application="fal-ai/qwen-image-edit/image-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Wan25PreviewImageToImage(FALNode):
    """
    Wan 2.5 Image to Image
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    prompt: str = Field(
        default="", description="The text prompt describing how to edit the image. Max 2000 characters."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate. Values from 1 to 4."
    )
    image_size: str = Field(
        default="square", description="The size of the generated image. Width and height must be between 384 and 1440 pixels."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    image_urls: list[str] = Field(
        default=[], description="URLs of images to edit. For single-image editing, provide 1 URL. For multi-reference generation, provide up to 2 URLs. If more than 2 URLs are provided, only the first 2 will be used."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to describe content to avoid. Max 500 characters."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-25-preview/image-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEditPlus(FALNode):
    """
    Qwen Image Edit Plus
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class Acceleration(Enum):
        """
        Acceleration level for image generation. Options: 'none', 'regular'. Higher acceleration increases speed. 'regular' balances speed and quality.
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
        default="", description="The prompt to generate the image with"
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="square_hd", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. Options: 'none', 'regular'. Higher acceleration increases speed. 'regular' balances speed and quality."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_scale: float = Field(
        default=4, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images to edit."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    num_inference_steps: int = Field(
        default=50, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "image_urls": self.image_urls,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit-plus",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class SeedvrUpscaleImage(FALNode):
    """
    SeedVR2
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class UpscaleMode(Enum):
        """
        The mode to use for the upscale. If 'target', the upscale factor will be calculated based on the target resolution. If 'factor', the upscale factor will be used directly.
        """
        TARGET = "target"
        FACTOR = "factor"

    class OutputFormat(Enum):
        """
        The format of the output image.
        """
        PNG = "png"
        JPG = "jpg"
        WEBP = "webp"

    class TargetResolution(Enum):
        """
        The target resolution to upscale to when `upscale_mode` is `target`.
        """
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"
        VALUE_1440P = "1440p"
        VALUE_2160P = "2160p"


    upscale_mode: UpscaleMode = Field(
        default=UpscaleMode.FACTOR, description="The mode to use for the upscale. If 'target', the upscale factor will be calculated based on the target resolution. If 'factor', the upscale factor will be used directly."
    )
    noise_scale: float = Field(
        default=0.1, description="The noise scale to use for the generation process."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPG, description="The format of the output image."
    )
    target_resolution: TargetResolution = Field(
        default=TargetResolution.VALUE_1080P, description="The target resolution to upscale to when `upscale_mode` is `target`."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The input image to be processed"
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

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "upscale_mode": self.upscale_mode.value,
            "noise_scale": self.noise_scale,
            "output_format": self.output_format.value,
            "target_resolution": self.target_resolution.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "upscale_factor": self.upscale_factor,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/seedvr/upscale/image",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ImageAppsV2ProductHolding(FALNode):
    """
    Product Holding
    editing, transformation, image-to-image, img2img, professional

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    aspect_ratio: str = Field(
        default="", description="Aspect ratio for 4K output"
    )
    product_image_url: ImageRef = Field(
        default=ImageRef(), description="Image URL of the product to be held by the person"
    )
    person_image_url: ImageRef = Field(
        default=ImageRef(), description="Image URL of the person who will hold the product"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        product_image_url_base64 = await context.image_to_base64(self.product_image_url)
        person_image_url_base64 = await context.image_to_base64(self.person_image_url)
        arguments = {
            "aspect_ratio": self.aspect_ratio,
            "product_image_url": f"data:image/png;base64,{product_image_url_base64}",
            "person_image_url": f"data:image/png;base64,{person_image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/image-apps-v2/product-holding",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ImageAppsV2ProductPhotography(FALNode):
    """
    Product Photography
    editing, transformation, image-to-image, img2img, professional

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    aspect_ratio: str = Field(
        default="", description="Aspect ratio for 4K output"
    )
    product_image_url: ImageRef = Field(
        default=ImageRef(), description="Image URL of the product to create professional studio photography"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        product_image_url_base64 = await context.image_to_base64(self.product_image_url)
        arguments = {
            "aspect_ratio": self.aspect_ratio,
            "product_image_url": f"data:image/png;base64,{product_image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/image-apps-v2/product-photography",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ImageAppsV2VirtualTryOn(FALNode):
    """
    Virtual Try-on
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    preserve_pose: bool = Field(
        default=True
    )
    aspect_ratio: str = Field(
        default="", description="Aspect ratio for 4K output (default: 3:4 for fashion)"
    )
    clothing_image_url: ImageRef = Field(
        default=ImageRef(), description="Clothing photo URL"
    )
    person_image_url: ImageRef = Field(
        default=ImageRef(), description="Person photo URL"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        clothing_image_url_base64 = await context.image_to_base64(self.clothing_image_url)
        person_image_url_base64 = await context.image_to_base64(self.person_image_url)
        arguments = {
            "preserve_pose": self.preserve_pose,
            "aspect_ratio": self.aspect_ratio,
            "clothing_image_url": f"data:image/png;base64,{clothing_image_url_base64}",
            "person_image_url": f"data:image/png;base64,{person_image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/image-apps-v2/virtual-try-on",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ImageAppsV2TextureTransform(FALNode):
    """
    Texture Transform
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class TargetTexture(Enum):
        COTTON = "cotton"
        DENIM = "denim"
        WOOL = "wool"
        FELT = "felt"
        WOOD = "wood"
        LEATHER = "leather"
        VELVET = "velvet"
        STONE = "stone"
        MARBLE = "marble"
        CERAMIC = "ceramic"
        CONCRETE = "concrete"
        BRICK = "brick"
        CLAY = "clay"
        FOAM = "foam"
        GLASS = "glass"
        METAL = "metal"
        SILK = "silk"
        FABRIC = "fabric"
        CRYSTAL = "crystal"
        RUBBER = "rubber"
        PLASTIC = "plastic"
        LACE = "lace"


    target_texture: TargetTexture = Field(
        default=TargetTexture.MARBLE
    )
    aspect_ratio: str = Field(
        default="", description="Aspect ratio for 4K output"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="Image URL for texture transformation"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "target_texture": self.target_texture.value,
            "aspect_ratio": self.aspect_ratio,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/image-apps-v2/texture-transform",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ImageAppsV2Relighting(FALNode):
    """
    Relighting
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class LightingStyle(Enum):
        NATURAL = "natural"
        STUDIO = "studio"
        GOLDEN_HOUR = "golden_hour"
        BLUE_HOUR = "blue_hour"
        DRAMATIC = "dramatic"
        SOFT = "soft"
        HARD = "hard"
        BACKLIGHT = "backlight"
        SIDE_LIGHT = "side_light"
        FRONT_LIGHT = "front_light"
        RIM_LIGHT = "rim_light"
        SUNSET = "sunset"
        SUNRISE = "sunrise"
        NEON = "neon"
        CANDLELIGHT = "candlelight"
        MOONLIGHT = "moonlight"
        SPOTLIGHT = "spotlight"
        AMBIENT = "ambient"


    aspect_ratio: str = Field(
        default="", description="Aspect ratio for 4K output"
    )
    lighting_style: LightingStyle = Field(
        default=LightingStyle.NATURAL
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="Image URL for relighting"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "aspect_ratio": self.aspect_ratio,
            "lighting_style": self.lighting_style.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/image-apps-v2/relighting",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ImageAppsV2StyleTransfer(FALNode):
    """
    Style Transfer
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class TargetStyle(Enum):
        ANIME_CHARACTER = "anime_character"
        CARTOON_3D = "cartoon_3d"
        HAND_DRAWN_ANIMATION = "hand_drawn_animation"
        CYBERPUNK_FUTURE = "cyberpunk_future"
        ANIME_GAME_STYLE = "anime_game_style"
        COMIC_BOOK_ANIMATION = "comic_book_animation"
        ANIMATED_SERIES = "animated_series"
        CARTOON_ANIMATION = "cartoon_animation"
        LOFI_AESTHETIC = "lofi_aesthetic"
        COTTAGECORE = "cottagecore"
        DARK_ACADEMIA = "dark_academia"
        Y2K = "y2k"
        VAPORWAVE = "vaporwave"
        LIMINAL_SPACE = "liminal_space"
        WEIRDCORE = "weirdcore"
        DREAMCORE = "dreamcore"
        SYNTHWAVE = "synthwave"
        OUTRUN = "outrun"
        PHOTOREALISTIC = "photorealistic"
        HYPERREALISTIC = "hyperrealistic"
        DIGITAL_ART = "digital_art"
        CONCEPT_ART = "concept_art"
        IMPRESSIONIST = "impressionist"
        ANIME = "anime"
        PIXEL_ART = "pixel_art"
        CLAYMATION = "claymation"


    target_style: TargetStyle = Field(
        default=TargetStyle.IMPRESSIONIST
    )
    aspect_ratio: str = Field(
        default="", description="Aspect ratio for 4K output"
    )
    style_reference_image_url: ImageRef = Field(
        default=ImageRef(), description="Optional reference image URL. When provided, the style will be inferred from this image instead of the selected preset style."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="Image URL for style transfer"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        style_reference_image_url_base64 = await context.image_to_base64(self.style_reference_image_url)
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "target_style": self.target_style.value,
            "aspect_ratio": self.aspect_ratio,
            "style_reference_image_url": f"data:image/png;base64,{style_reference_image_url_base64}",
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/image-apps-v2/style-transfer",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ImageAppsV2PhotoRestoration(FALNode):
    """
    Photo Restoration
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    enhance_resolution: bool = Field(
        default=True
    )
    aspect_ratio: str = Field(
        default="", description="Aspect ratio for 4K output (default: 4:3 for classic photos)"
    )
    remove_scratches: bool = Field(
        default=True
    )
    fix_colors: bool = Field(
        default=True
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="Old or damaged photo URL to restore"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "enhance_resolution": self.enhance_resolution,
            "aspect_ratio": self.aspect_ratio,
            "remove_scratches": self.remove_scratches,
            "fix_colors": self.fix_colors,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/image-apps-v2/photo-restoration",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ImageAppsV2PortraitEnhance(FALNode):
    """
    Portrait Enhance
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    aspect_ratio: str = Field(
        default="", description="Aspect ratio for 4K output (default: 3:4 for portraits)"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="Portrait image URL to enhance"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "aspect_ratio": self.aspect_ratio,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/image-apps-v2/portrait-enhance",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ImageAppsV2PhotographyEffects(FALNode):
    """
    Photography Effects
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class EffectType(Enum):
        FILM = "film"
        VINTAGE_FILM = "vintage_film"
        PORTRAIT_PHOTOGRAPHY = "portrait_photography"
        FASHION_PHOTOGRAPHY = "fashion_photography"
        STREET_PHOTOGRAPHY = "street_photography"
        SEPIA_TONE = "sepia_tone"
        FILM_GRAIN = "film_grain"
        LIGHT_LEAKS = "light_leaks"
        VIGNETTE_EFFECT = "vignette_effect"
        INSTANT_CAMERA = "instant_camera"
        GOLDEN_HOUR = "golden_hour"
        DRAMATIC_LIGHTING = "dramatic_lighting"
        SOFT_FOCUS = "soft_focus"
        BOKEH_EFFECT = "bokeh_effect"
        HIGH_CONTRAST = "high_contrast"
        DOUBLE_EXPOSURE = "double_exposure"


    effect_type: EffectType = Field(
        default=EffectType.FILM
    )
    aspect_ratio: str = Field(
        default="", description="Aspect ratio for 4K output"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="Image URL for photography effects"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "effect_type": self.effect_type.value,
            "aspect_ratio": self.aspect_ratio,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/image-apps-v2/photography-effects",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ImageAppsV2Perspective(FALNode):
    """
    Perspective Change
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class TargetPerspective(Enum):
        FRONT = "front"
        LEFT_SIDE = "left_side"
        RIGHT_SIDE = "right_side"
        BACK = "back"
        TOP_DOWN = "top_down"
        BOTTOM_UP = "bottom_up"
        BIRDS_EYE = "birds_eye"
        THREE_QUARTER_LEFT = "three_quarter_left"
        THREE_QUARTER_RIGHT = "three_quarter_right"


    aspect_ratio: str = Field(
        default="", description="Aspect ratio for 4K output"
    )
    target_perspective: TargetPerspective = Field(
        default=TargetPerspective.FRONT
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="Image URL for perspective change"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "aspect_ratio": self.aspect_ratio,
            "target_perspective": self.target_perspective.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/image-apps-v2/perspective",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ImageAppsV2ObjectRemoval(FALNode):
    """
    Object Removal
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    aspect_ratio: str = Field(
        default="", description="Aspect ratio for 4K output"
    )
    object_to_remove: str = Field(
        default="", description="Object to remove"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="Image URL containing object to remove"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "aspect_ratio": self.aspect_ratio,
            "object_to_remove": self.object_to_remove,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/image-apps-v2/object-removal",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ImageAppsV2HeadshotPhoto(FALNode):
    """
    Headshot Generator
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class BackgroundStyle(Enum):
        PROFESSIONAL = "professional"
        CORPORATE = "corporate"
        CLEAN = "clean"
        GRADIENT = "gradient"


    aspect_ratio: str = Field(
        default="", description="Aspect ratio for 4K output (default: 3:4 for portraits)"
    )
    background_style: BackgroundStyle = Field(
        default=BackgroundStyle.PROFESSIONAL
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="Portrait image URL to convert to professional headshot"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "aspect_ratio": self.aspect_ratio,
            "background_style": self.background_style.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/image-apps-v2/headshot-photo",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ImageAppsV2HairChange(FALNode):
    """
    Hair Change
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class TargetHairstyle(Enum):
        SHORT_HAIR = "short_hair"
        MEDIUM_LONG_HAIR = "medium_long_hair"
        LONG_HAIR = "long_hair"
        CURLY_HAIR = "curly_hair"
        WAVY_HAIR = "wavy_hair"
        HIGH_PONYTAIL = "high_ponytail"
        BUN = "bun"
        BOB_CUT = "bob_cut"
        PIXIE_CUT = "pixie_cut"
        BRAIDS = "braids"
        STRAIGHT_HAIR = "straight_hair"
        AFRO = "afro"
        DREADLOCKS = "dreadlocks"
        BUZZ_CUT = "buzz_cut"
        MOHAWK = "mohawk"
        BANGS = "bangs"
        SIDE_PART = "side_part"
        MIDDLE_PART = "middle_part"

    class HairColor(Enum):
        BLACK = "black"
        DARK_BROWN = "dark_brown"
        LIGHT_BROWN = "light_brown"
        BLONDE = "blonde"
        PLATINUM_BLONDE = "platinum_blonde"
        RED = "red"
        AUBURN = "auburn"
        GRAY = "gray"
        SILVER = "silver"
        BLUE = "blue"
        GREEN = "green"
        PURPLE = "purple"
        PINK = "pink"
        RAINBOW = "rainbow"
        NATURAL = "natural"
        HIGHLIGHTS = "highlights"
        OMBRE = "ombre"
        BALAYAGE = "balayage"


    target_hairstyle: TargetHairstyle = Field(
        default=TargetHairstyle.LONG_HAIR
    )
    aspect_ratio: str = Field(
        default="", description="Aspect ratio for 4K output (default: 3:4 for portraits)"
    )
    hair_color: HairColor = Field(
        default=HairColor.NATURAL
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="Portrait image URL for hair change"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "target_hairstyle": self.target_hairstyle.value,
            "aspect_ratio": self.aspect_ratio,
            "hair_color": self.hair_color.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/image-apps-v2/hair-change",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ImageAppsV2ExpressionChange(FALNode):
    """
    Expression Change
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class TargetExpression(Enum):
        SMILE = "smile"
        SURPRISE = "surprise"
        GLARE = "glare"
        PANIC = "panic"
        SHYNESS = "shyness"
        LAUGH = "laugh"
        CRY = "cry"
        ANGRY = "angry"
        SAD = "sad"
        HAPPY = "happy"
        EXCITED = "excited"
        SHOCKED = "shocked"
        CONFUSED = "confused"
        FOCUSED = "focused"
        DREAMY = "dreamy"
        SERIOUS = "serious"
        PLAYFUL = "playful"
        MYSTERIOUS = "mysterious"
        CONFIDENT = "confident"
        THOUGHTFUL = "thoughtful"


    aspect_ratio: str = Field(
        default="", description="Aspect ratio for 4K output (default: 3:4 for portraits)"
    )
    target_expression: TargetExpression = Field(
        default=TargetExpression.SMILE
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="Portrait image URL for expression change"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "aspect_ratio": self.aspect_ratio,
            "target_expression": self.target_expression.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/image-apps-v2/expression-change",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ImageAppsV2CityTeleport(FALNode):
    """
    City Teleport
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class PhotoShot(Enum):
        """
        Type of photo shot
        """
        EXTREME_CLOSE_UP = "extreme_close_up"
        CLOSE_UP = "close_up"
        MEDIUM_CLOSE_UP = "medium_close_up"
        MEDIUM_SHOT = "medium_shot"
        MEDIUM_LONG_SHOT = "medium_long_shot"
        LONG_SHOT = "long_shot"
        EXTREME_LONG_SHOT = "extreme_long_shot"
        FULL_BODY = "full_body"

    class CameraAngle(Enum):
        """
        Camera angle for the shot
        """
        EYE_LEVEL = "eye_level"
        LOW_ANGLE = "low_angle"
        HIGH_ANGLE = "high_angle"
        DUTCH_ANGLE = "dutch_angle"
        BIRDS_EYE_VIEW = "birds_eye_view"
        WORMS_EYE_VIEW = "worms_eye_view"
        OVERHEAD = "overhead"
        SIDE_ANGLE = "side_angle"


    city_image_url: ImageRef = Field(
        default=ImageRef(), description="Optional city background image URL. When provided, the person will be blended into this custom scene."
    )
    aspect_ratio: str = Field(
        default="", description="Aspect ratio for 4K output"
    )
    city_name: str = Field(
        default="", description="City name (used when city_image_url is not provided)"
    )
    photo_shot: PhotoShot = Field(
        default=PhotoShot.MEDIUM_SHOT, description="Type of photo shot"
    )
    camera_angle: CameraAngle = Field(
        default=CameraAngle.EYE_LEVEL, description="Camera angle for the shot"
    )
    person_image_url: ImageRef = Field(
        default=ImageRef(), description="Person photo URL"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        city_image_url_base64 = await context.image_to_base64(self.city_image_url)
        person_image_url_base64 = await context.image_to_base64(self.person_image_url)
        arguments = {
            "city_image_url": f"data:image/png;base64,{city_image_url_base64}",
            "aspect_ratio": self.aspect_ratio,
            "city_name": self.city_name,
            "photo_shot": self.photo_shot.value,
            "camera_angle": self.camera_angle.value,
            "person_image_url": f"data:image/png;base64,{person_image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/image-apps-v2/city-teleport",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ImageAppsV2AgeModify(FALNode):
    """
    Age Modify
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    image_url: ImageRef = Field(
        default=ImageRef(), description="Portrait image URL for age modification"
    )
    aspect_ratio: str = Field(
        default="", description="Aspect ratio for 4K output (default: 3:4 for portraits)"
    )
    preserve_identity: bool = Field(
        default=True
    )
    target_age: int = Field(
        default=30
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "aspect_ratio": self.aspect_ratio,
            "preserve_identity": self.preserve_identity,
            "target_age": self.target_age,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/image-apps-v2/age-modify",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ImageAppsV2MakeupApplication(FALNode):
    """
    Makeup Changer
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class Intensity(Enum):
        LIGHT = "light"
        MEDIUM = "medium"
        HEAVY = "heavy"
        DRAMATIC = "dramatic"

    class MakeupStyle(Enum):
        NATURAL = "natural"
        GLAMOROUS = "glamorous"
        SMOKY_EYES = "smoky_eyes"
        BOLD_LIPS = "bold_lips"
        NO_MAKEUP = "no_makeup"
        REMOVE_MAKEUP = "remove_makeup"
        DRAMATIC = "dramatic"
        BRIDAL = "bridal"
        PROFESSIONAL = "professional"
        KOREAN_STYLE = "korean_style"
        ARTISTIC = "artistic"


    aspect_ratio: str = Field(
        default="", description="Aspect ratio for 4K output (default: 3:4 for portraits)"
    )
    intensity: Intensity = Field(
        default=Intensity.MEDIUM
    )
    makeup_style: MakeupStyle = Field(
        default=MakeupStyle.NATURAL
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="Portrait image URL for makeup application"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "aspect_ratio": self.aspect_ratio,
            "intensity": self.intensity.value,
            "makeup_style": self.makeup_style.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/image-apps-v2/makeup-application",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEditInpaint(FALNode):
    """
    Qwen Image Edit
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class Acceleration(Enum):
        """
        Acceleration level for image generation. Options: 'none', 'regular'. Higher acceleration increases speed. 'regular' balances speed and quality.
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
    image_size: str = Field(
        default="", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. Options: 'none', 'regular'. Higher acceleration increases speed. 'regular' balances speed and quality."
    )
    guidance_scale: float = Field(
        default=4, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to edit."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    strength: float = Field(
        default=0.93, description="Strength of noising process for inpainting"
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    mask_url: str = Field(
        default="", description="The URL of the mask for inpainting"
    )
    num_inference_steps: int = Field(
        default=30, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "num_images": self.num_images,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "strength": self.strength,
            "seed": self.seed,
            "mask_url": self.mask_url,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-edit/inpaint",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class FluxSrpoImageToImage(FALNode):
    """
    FLUX.1 SRPO [dev]
    flux, editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="The speed of the generation. The higher the speed, the faster the generation."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to generate an image from."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    strength: float = Field(
        default=0.95, description="The strength of the initial image. Higher strength values are better for this model."
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
        default=40, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "strength": self.strength,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux/srpo/image-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Flux1SrpoImageToImage(FALNode):
    """
    FLUX.1 SRPO [dev]
    flux, editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The speed of the generation. The higher the speed, the faster the generation."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to generate an image from."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    strength: float = Field(
        default=0.95, description="The strength of the initial image. Higher strength values are better for this model."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to perform."
    )
    seed: str = Field(
        default="", description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    guidance_scale: float = Field(
        default=4.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "strength": self.strength,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-1/srpo/image-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEditLora(FALNode):
    """
    Qwen Image Edit Lora
    editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class Acceleration(Enum):
        """
        Acceleration level for image generation. Options: 'none', 'regular'. Higher acceleration increases speed. 'regular' balances speed and quality.
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
        default="", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="Acceleration level for image generation. Options: 'none', 'regular'. Higher acceleration increases speed. 'regular' balances speed and quality."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to edit."
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the image to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the image directly in the response without going through the CDN."
    )
    loras: list[str] = Field(
        default=[], description="The LoRAs to use for the image generation. You can use up to 3 LoRAs and they will be merged together to generate the final image."
    )
    guidance_scale: float = Field(
        default=4, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=30, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "loras": self.loras,
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
            application="fal-ai/qwen-image-edit-lora",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ViduReferenceToImage(FALNode):
    """
    Vidu
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
    reference_image_urls: list[str] = Field(
        default=[], description="URLs of the reference images to use for consistent subject appearance"
    )
    seed: int = Field(
        default=-1, description="Random seed for generation"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "reference_image_urls": self.reference_image_urls,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/vidu/reference-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class BytedanceSeedreamV4Edit(FALNode):
    """
    Bytedance Seedream v4 Edit
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class EnhancePromptMode(Enum):
        """
        The mode to use for enhancing prompt enhancement. Standard mode provides higher quality results but takes longer to generate. Fast mode provides average quality results but takes less time to generate.
        """
        STANDARD = "standard"
        FAST = "fast"


    prompt: str = Field(
        default="", description="The text prompt used to edit the image"
    )
    num_images: int = Field(
        default=1, description="Number of separate model generations to be run with the prompt."
    )
    image_size: str = Field(
        default="", description="The size of the generated image. The minimum total image area is 921600 pixels. Failing this, the image size will be adjusted to by scaling it up, while maintaining the aspect ratio."
    )
    max_images: int = Field(
        default=1, description="If set to a number greater than one, enables multi-image generation. The model will potentially return up to `max_images` images every generation, and in total, `num_images` generations will be carried out. In total, the number of images generated will be between `num_images` and `max_images*num_images`. The total number of images (image inputs + image outputs) must not exceed 15"
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
    image_urls: list[str] = Field(
        default=[], description="List of URLs of input images for editing. Presently, up to 10 image inputs are allowed. If over 10 images are sent, only the last 10 will be used."
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
            "image_urls": self.image_urls,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/bytedance/seedream/v4/edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class WanV2_2A14BImageToImage(FALNode):
    """
    Wan
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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

    class AspectRatio(Enum):
        """
        Aspect ratio of the generated image. If 'auto', the aspect ratio will be determined automatically based on the input image.
        """
        AUTO = "auto"
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"


    shift: float = Field(
        default=2
    )
    prompt: str = Field(
        default="", description="The text prompt to guide image generation."
    )
    image_size: str = Field(
        default=""
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level to use. The more acceleration, the faster the generation, but with lower quality. The recommended value is 'regular'."
    )
    guidance_scale: float = Field(
        default=3.5, description="Classifier-free guidance scale."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, input data will be checked for safety before processing."
    )
    image_format: ImageFormat = Field(
        default=ImageFormat.JPEG, description="The format of the output image."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt for video generation."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.AUTO, description="Aspect ratio of the generated image. If 'auto', the aspect ratio will be determined automatically based on the input image."
    )
    enable_output_safety_checker: bool = Field(
        default=False, description="If set to true, output video will be checked for safety after generation."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the input image."
    )
    strength: float = Field(
        default=0.5, description="Denoising strength. 1.0 = fully remake; 0.0 = preserve original."
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
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "shift": self.shift,
            "prompt": self.prompt,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "image_format": self.image_format.value,
            "negative_prompt": self.negative_prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "enable_output_safety_checker": self.enable_output_safety_checker,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "strength": self.strength,
            "guidance_scale_2": self.guidance_scale_2,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan/v2.2-a14b/image-to-image",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Uso(FALNode):
    """
    Uso
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class OutputFormat(Enum):
        """
        Output image format. PNG preserves transparency, JPEG is smaller.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="Text prompt for generation. Can be empty for pure style transfer."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate in parallel."
    )
    image_size: str = Field(
        default="square_hd", description="The size of the generated image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="Output image format. PNG preserves transparency, JPEG is smaller."
    )
    keep_size: bool = Field(
        default=False, description="Preserve the layout and dimensions of the input content image. Useful for style transfer."
    )
    input_image_urls: list[str] = Field(
        default=[], description="List of image URLs in order: [content_image, style_image, extra_style_image]."
    )
    sync_mode: bool = Field(
        default=False, description="If true, wait for generation and upload before returning. Increases latency but provides immediate access to images."
    )
    guidance_scale: float = Field(
        default=4, description="How closely to follow the prompt. Higher values stick closer to the prompt."
    )
    num_inference_steps: int = Field(
        default=28, description="Number of denoising steps. More steps can improve quality but increase generation time."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducible generation. Use same seed for consistent results."
    )
    negative_prompt: str = Field(
        default="", description="What you don't want in the image. Use it to exclude unwanted elements, styles, or artifacts."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Enable NSFW content detection and filtering."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "keep_size": self.keep_size,
            "input_image_urls": self.input_image_urls,
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
            application="fal-ai/uso",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Gemini25FlashImageEdit(FALNode):
    """
    Gemini 2.5 Flash Image
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated image.
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


    prompt: str = Field(
        default="", description="The prompt for image editing."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.AUTO, description="The aspect ratio of the generated image."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images to use for image-to-image generation or image editing."
    )
    limit_generations: bool = Field(
        default=False, description="Experimental parameter to limit the number of generations from each round of prompting to 1. Set to `True` to to disregard any instructions in the prompt regarding the number of images to generate."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "num_images": self.num_images,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "image_urls": self.image_urls,
            "limit_generations": self.limit_generations,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/gemini-25-flash-image/edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageImageToImage(FALNode):
    """
    Qwen Image
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="Acceleration level for image generation. Options: 'none', 'regular', 'high'. Higher acceleration increases speed. 'regular' balances speed and quality. 'high' is recommended for images without text."
    )
    image_size: str = Field(
        default="", description="The size of the generated image. By default, we will use the provided image for determining the image_size."
    )
    loras: list[str] = Field(
        default=[], description="The LoRAs to use for the image generation. You can use up to 3 LoRAs and they will be merged together to generate the final image."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    guidance_scale: float = Field(
        default=2.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    use_turbo: bool = Field(
        default=False, description="Enable turbo mode for faster generation with high quality. When enabled, uses optimized settings (10 steps, CFG=1.2)."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The reference image to guide the generation."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    strength: float = Field(
        default=0.6, description="Denoising strength. 1.0 = fully remake; 0.0 = preserve original."
    )
    num_inference_steps: int = Field(
        default=30, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "acceleration": self.acceleration.value,
            "image_size": self.image_size,
            "loras": self.loras,
            "enable_safety_checker": self.enable_safety_checker,
            "guidance_scale": self.guidance_scale,
            "use_turbo": self.use_turbo,
            "negative_prompt": self.negative_prompt,
            "num_images": self.num_images,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "strength": self.strength,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image/image-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class BriaReimagine3_2(FALNode):
    """
    Reimagine
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
    depth_preprocess: bool = Field(
        default=True, description="Depth image preprocess."
    )
    canny_preprocess: bool = Field(
        default=True, description="Canny image preprocess."
    )
    depth_image_url: ImageRef = Field(
        default="", description="Depth control image (file or URL)."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for text."
    )
    canny_image_url: ImageRef = Field(
        default="", description="Canny edge control image (file or URL)."
    )
    negative_prompt: str = Field(
        default="Logo,Watermark,Ugly,Morbid,Extra fingers,Poorly drawn hands,Mutation,Blurry,Extra limbs,Gross proportions,Missing arms,Mutated hands,Long neck,Duplicate,Mutilated,Mutilated hands,Poorly drawn face,Deformed,Bad anatomy,Cloned face,Malformed limbs,Missing legs,Too many fingers", description="Negative prompt for image generation."
    )
    depth_scale: float = Field(
        default=0.5, description="Depth control strength (0.0 to 1.0)."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1, description="Aspect ratio. Options: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9"
    )
    sync_mode: bool = Field(
        default=False, description="If true, returns the image directly in the response (increases latency)."
    )
    prompt_enhancer: bool = Field(
        default=True, description="Whether to improve the prompt."
    )
    truncate_prompt: bool = Field(
        default=True, description="Whether to truncate the prompt."
    )
    seed: int = Field(
        default=5555, description="Random seed for reproducibility."
    )
    canny_scale: float = Field(
        default=0.5, description="Canny edge control strength (0.0 to 1.0)."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of inference steps."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        depth_image_url_base64 = await context.image_to_base64(self.depth_image_url)
        canny_image_url_base64 = await context.image_to_base64(self.canny_image_url)
        arguments = {
            "prompt": self.prompt,
            "depth_preprocess": self.depth_preprocess,
            "canny_preprocess": self.canny_preprocess,
            "depth_image_url": f"data:image/png;base64,{depth_image_url_base64}",
            "guidance_scale": self.guidance_scale,
            "canny_image_url": f"data:image/png;base64,{canny_image_url_base64}",
            "negative_prompt": self.negative_prompt,
            "depth_scale": self.depth_scale,
            "aspect_ratio": self.aspect_ratio.value,
            "sync_mode": self.sync_mode,
            "prompt_enhancer": self.prompt_enhancer,
            "truncate_prompt": self.truncate_prompt,
            "seed": self.seed,
            "canny_scale": self.canny_scale,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/reimagine/3.2",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class NanoBananaEdit(FALNode):
    """
    Nano Banana
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated image.
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


    prompt: str = Field(
        default="", description="The prompt for image editing."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.AUTO, description="The aspect ratio of the generated image."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    image_urls: list[str] = Field(
        default=[], description="The URLs of the images to use for image-to-image generation or image editing."
    )
    limit_generations: bool = Field(
        default=False, description="Experimental parameter to limit the number of generations from each round of prompting to 1. Set to `True` to to disregard any instructions in the prompt regarding the number of images to generate."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "num_images": self.num_images,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "image_urls": self.image_urls,
            "limit_generations": self.limit_generations,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/nano-banana/edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Nextstep1(FALNode):
    """
    Nextstep 1
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    prompt: str = Field(
        default="", description="The prompt to edit the image."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to edit."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/nextstep-1",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class QwenImageEdit(FALNode):
    """
    Qwen Image Edit
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class Acceleration(Enum):
        """
        Acceleration level for image generation. Options: 'none', 'regular'. Higher acceleration increases speed. 'regular' balances speed and quality.
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
        default="", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. Options: 'none', 'regular'. Higher acceleration increases speed. 'regular' balances speed and quality."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to edit."
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
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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
            application="fal-ai/qwen-image-edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class IdeogramCharacterEdit(FALNode):
    """
    Ideogram V3 Character Edit
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class Style(Enum):
        """
        The style type to generate with. Cannot be used with style_codes.
        """
        AUTO = "AUTO"
        REALISTIC = "REALISTIC"
        FICTION = "FICTION"

    class RenderingSpeed(Enum):
        """
        The rendering speed to use.
        """
        TURBO = "TURBO"
        BALANCED = "BALANCED"
        QUALITY = "QUALITY"


    prompt: str = Field(
        default="", description="The prompt to fill the masked part of the image."
    )
    style: Style = Field(
        default=Style.AUTO, description="The style type to generate with. Cannot be used with style_codes."
    )
    expand_prompt: bool = Field(
        default=True, description="Determine if MagicPrompt should be used in generating the request or not."
    )
    rendering_speed: RenderingSpeed = Field(
        default=RenderingSpeed.BALANCED, description="The rendering speed to use."
    )
    reference_mask_urls: list[str] = Field(
        default=[], description="A set of masks to apply to the character references. Currently only 1 mask is supported, rest will be ignored. (maximum total size 10MB across all character references). The masks should be in JPEG, PNG or WebP format"
    )
    reference_image_urls: list[str] = Field(
        default=[], description="A set of images to use as character references. Currently only 1 image is supported, rest will be ignored. (maximum total size 10MB across all character references). The images should be in JPEG, PNG or WebP format"
    )
    image_urls: ImageRef = Field(
        default=ImageRef(), description="A set of images to use as style references (maximum total size 10MB across all style references). The images should be in JPEG, PNG or WebP format"
    )
    num_images: int = Field(
        default=1, description="Number of images to generate."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The image URL to generate an image from. MUST have the exact same dimensions (width and height) as the mask image."
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
    mask_url: ImageRef = Field(
        default=ImageRef(), description="The mask URL to inpaint the image. MUST have the exact same dimensions (width and height) as the input image."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_urls_base64 = await context.image_to_base64(self.image_urls)
        image_url_base64 = await context.image_to_base64(self.image_url)
        mask_url_base64 = await context.image_to_base64(self.mask_url)
        arguments = {
            "prompt": self.prompt,
            "style": self.style.value,
            "expand_prompt": self.expand_prompt,
            "rendering_speed": self.rendering_speed.value,
            "reference_mask_urls": self.reference_mask_urls,
            "reference_image_urls": self.reference_image_urls,
            "image_urls": f"data:image/png;base64,{image_urls_base64}",
            "num_images": self.num_images,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "style_codes": self.style_codes,
            "color_palette": self.color_palette,
            "sync_mode": self.sync_mode,
            "seed": self.seed,
            "mask_url": f"data:image/png;base64,{mask_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ideogram/character/edit",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class IdeogramCharacter(FALNode):
    """
    Ideogram V3 Character
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class Style(Enum):
        """
        The style type to generate with. Cannot be used with style_codes.
        """
        AUTO = "AUTO"
        REALISTIC = "REALISTIC"
        FICTION = "FICTION"

    class RenderingSpeed(Enum):
        """
        The rendering speed to use.
        """
        TURBO = "TURBO"
        BALANCED = "BALANCED"
        QUALITY = "QUALITY"


    prompt: str = Field(
        default="", description="The prompt to fill the masked part of the image."
    )
    image_size: str = Field(
        default="square_hd", description="The resolution of the generated image"
    )
    style: Style = Field(
        default=Style.AUTO, description="The style type to generate with. Cannot be used with style_codes."
    )
    expand_prompt: bool = Field(
        default=True, description="Determine if MagicPrompt should be used in generating the request or not."
    )
    rendering_speed: RenderingSpeed = Field(
        default=RenderingSpeed.BALANCED, description="The rendering speed to use."
    )
    reference_mask_urls: list[str] = Field(
        default=[], description="A set of masks to apply to the character references. Currently only 1 mask is supported, rest will be ignored. (maximum total size 10MB across all character references). The masks should be in JPEG, PNG or WebP format"
    )
    reference_image_urls: list[str] = Field(
        default=[], description="A set of images to use as character references. Currently only 1 image is supported, rest will be ignored. (maximum total size 10MB across all character references). The images should be in JPEG, PNG or WebP format"
    )
    image_urls: ImageRef = Field(
        default=ImageRef(), description="A set of images to use as style references (maximum total size 10MB across all style references). The images should be in JPEG, PNG or WebP format"
    )
    negative_prompt: str = Field(
        default="", description="Description of what to exclude from an image. Descriptions in the prompt take precedence to descriptions in the negative prompt."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate."
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

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_urls_base64 = await context.image_to_base64(self.image_urls)
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size,
            "style": self.style.value,
            "expand_prompt": self.expand_prompt,
            "rendering_speed": self.rendering_speed.value,
            "reference_mask_urls": self.reference_mask_urls,
            "reference_image_urls": self.reference_image_urls,
            "image_urls": f"data:image/png;base64,{image_urls_base64}",
            "negative_prompt": self.negative_prompt,
            "num_images": self.num_images,
            "style_codes": self.style_codes,
            "color_palette": self.color_palette,
            "sync_mode": self.sync_mode,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ideogram/character",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class IdeogramCharacterRemix(FALNode):
    """
    Ideogram V3 Character Remix
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class Style(Enum):
        """
        The style type to generate with. Cannot be used with style_codes.
        """
        AUTO = "AUTO"
        REALISTIC = "REALISTIC"
        FICTION = "FICTION"

    class RenderingSpeed(Enum):
        """
        The rendering speed to use.
        """
        TURBO = "TURBO"
        BALANCED = "BALANCED"
        QUALITY = "QUALITY"


    prompt: str = Field(
        default="", description="The prompt to remix the image with"
    )
    image_size: str = Field(
        default="square_hd", description="The resolution of the generated image"
    )
    style: Style = Field(
        default=Style.AUTO, description="The style type to generate with. Cannot be used with style_codes."
    )
    expand_prompt: bool = Field(
        default=True, description="Determine if MagicPrompt should be used in generating the request or not."
    )
    rendering_speed: RenderingSpeed = Field(
        default=RenderingSpeed.BALANCED, description="The rendering speed to use."
    )
    reference_mask_urls: list[str] = Field(
        default=[], description="A set of masks to apply to the character references. Currently only 1 mask is supported, rest will be ignored. (maximum total size 10MB across all character references). The masks should be in JPEG, PNG or WebP format"
    )
    reference_image_urls: list[str] = Field(
        default=[], description="A set of images to use as character references. Currently only 1 image is supported, rest will be ignored. (maximum total size 10MB across all character references). The images should be in JPEG, PNG or WebP format"
    )
    image_urls: ImageRef = Field(
        default=ImageRef(), description="A set of images to use as style references (maximum total size 10MB across all style references). The images should be in JPEG, PNG or WebP format"
    )
    negative_prompt: str = Field(
        default="", description="Description of what to exclude from an image. Descriptions in the prompt take precedence to descriptions in the negative prompt."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The image URL to remix"
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
    strength: float = Field(
        default=0.8, description="Strength of the input image in the remix"
    )
    seed: str = Field(
        default="", description="Seed for the random number generator"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_urls_base64 = await context.image_to_base64(self.image_urls)
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size,
            "style": self.style.value,
            "expand_prompt": self.expand_prompt,
            "rendering_speed": self.rendering_speed.value,
            "reference_mask_urls": self.reference_mask_urls,
            "reference_image_urls": self.reference_image_urls,
            "image_urls": f"data:image/png;base64,{image_urls_base64}",
            "negative_prompt": self.negative_prompt,
            "num_images": self.num_images,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "style_codes": self.style_codes,
            "color_palette": self.color_palette,
            "sync_mode": self.sync_mode,
            "strength": self.strength,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ideogram/character/remix",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class FluxKreaLoraInpainting(FALNode):
    """
    FLUX.1 Krea [dev] Inpainting with LoRAs
    flux, editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
        default="", description="The size of the generated image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of image to use for inpainting. or img2img"
    )
    loras: list[str] = Field(
        default=[], description="The LoRAs to use for the image generation. You can use any number of LoRAs and they will be merged together to generate the final image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    strength: float = Field(
        default=0.85, description="The strength to use for inpainting/image-to-image. Only used if the image_url is provided. 1.0 is completely remakes the image while 0.0 preserves the original."
    )
    guidance_scale: float = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    mask_url: str = Field(
        default="", description="The mask to area to Inpaint in."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "loras": self.loras,
            "sync_mode": self.sync_mode,
            "strength": self.strength,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "mask_url": self.mask_url,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-krea-lora/inpainting",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class FluxKreaLoraImageToImage(FALNode):
    """
    FLUX.1 Krea [dev] with LoRAs
    flux, editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
        default="", description="The size of the generated image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of image to use for inpainting. or img2img"
    )
    loras: list[str] = Field(
        default=[], description="The LoRAs to use for the image generation. You can use any number of LoRAs and they will be merged together to generate the final image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    strength: float = Field(
        default=0.85, description="The strength to use for inpainting/image-to-image. Only used if the image_url is provided. 1.0 is completely remakes the image while 0.0 preserves the original."
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
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "loras": self.loras,
            "sync_mode": self.sync_mode,
            "strength": self.strength,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-krea-lora/image-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class FluxKreaImageToImage(FALNode):
    """
    FLUX.1 Krea [dev]
    flux, editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="The speed of the generation. The higher the speed, the faster the generation."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to generate an image from."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    strength: float = Field(
        default=0.95, description="The strength of the initial image. Higher strength values are better for this model."
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
        default=40, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "strength": self.strength,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux/krea/image-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class FluxKreaRedux(FALNode):
    """
    FLUX.1 Krea [dev] Redux
    flux, editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to generate an image from."
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
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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
            application="fal-ai/flux/krea/redux",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Flux1KreaImageToImage(FALNode):
    """
    FLUX.1 Krea [dev]
    flux, editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The speed of the generation. The higher the speed, the faster the generation."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to generate an image from."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    strength: float = Field(
        default=0.95, description="The strength of the initial image. Higher strength values are better for this model."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to perform."
    )
    seed: str = Field(
        default="", description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    guidance_scale: float = Field(
        default=4.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "strength": self.strength,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-1/krea/image-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Flux1KreaRedux(FALNode):
    """
    FLUX.1 Krea [dev] Redux
    flux, editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to generate an image from."
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
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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
            application="fal-ai/flux-1/krea/redux",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class FluxKontextLoraInpaint(FALNode):
    """
    Flux Kontext Lora
    flux, editing, transformation, image-to-image, img2img, lora

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
        default="", description="The prompt for the image to image task."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="The speed of the generation. The higher the speed, the faster the generation."
    )
    reference_image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the reference image for inpainting."
    )
    loras: list[str] = Field(
        default=[], description="The LoRAs to use for the image generation. You can use any number of LoRAs and they will be merged together to generate the final image."
    )
    guidance_scale: float = Field(
        default=2.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to be inpainted."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    strength: float = Field(
        default=0.88, description="The strength of the initial image. Higher strength values are better for this model."
    )
    num_inference_steps: int = Field(
        default=30, description="The number of inference steps to perform."
    )
    mask_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the mask for inpainting."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        reference_image_url_base64 = await context.image_to_base64(self.reference_image_url)
        image_url_base64 = await context.image_to_base64(self.image_url)
        mask_url_base64 = await context.image_to_base64(self.mask_url)
        arguments = {
            "prompt": self.prompt,
            "acceleration": self.acceleration.value,
            "reference_image_url": f"data:image/png;base64,{reference_image_url_base64}",
            "loras": self.loras,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "num_images": self.num_images,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "strength": self.strength,
            "num_inference_steps": self.num_inference_steps,
            "mask_url": f"data:image/png;base64,{mask_url_base64}",
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-kontext-lora/inpaint",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Hunyuan_World(FALNode):
    """
    Hunyuan World
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    prompt: str = Field(
        default="", description="The prompt to use for the panorama generation."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to convert to a panorama."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan_world",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ImageEditingRetouch(FALNode):
    """
    Image Editing
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    lora_scale: float = Field(
        default=1, description="The scale factor for the LoRA model. Controls the strength of the LoRA effect."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image to retouch."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_scale: float = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of inference steps for sampling."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "lora_scale": self.lora_scale,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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
            application="fal-ai/image-editing/retouch",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class HidreamE11(FALNode):
    """
    Hidream E1 1
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
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
    image_guidance_scale: float = Field(
        default=2, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your initial image when looking for a related image to show you."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of an input image to edit."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_scale: float = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=50, description="The number of inference steps to perform."
    )
    target_image_description: str = Field(
        default="", description="The description of the target image after your edits have been made. Leave this blank to allow the model to use its own imagination."
    )
    negative_prompt: str = Field(
        default="low resolution, blur", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_guidance_scale": self.image_guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "target_image_description": self.target_image_description,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/hidream-e1-1",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Rife(FALNode):
    """
    RIFE
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class OutputFormat(Enum):
        """
        The format of the output images. Only applicable if output_type is 'images'.
        """
        PNG = "png"
        JPEG = "jpeg"

    class OutputType(Enum):
        """
        The type of output to generate; either individual images or a video.
        """
        IMAGES = "images"
        VIDEO = "video"


    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the output images. Only applicable if output_type is 'images'."
    )
    fps: int = Field(
        default=8, description="Frames per second for the output video. Only applicable if output_type is 'video'."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    include_end: bool = Field(
        default=False, description="Whether to include the end image in the output."
    )
    include_start: bool = Field(
        default=False, description="Whether to include the start image in the output."
    )
    num_frames: int = Field(
        default=1, description="The number of frames to generate between the input images."
    )
    end_image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the second image to use as the ending point for interpolation."
    )
    output_type: OutputType = Field(
        default=OutputType.IMAGES, description="The type of output to generate; either individual images or a video."
    )
    start_image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the first image to use as the starting point for interpolation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        end_image_url_base64 = await context.image_to_base64(self.end_image_url)
        start_image_url_base64 = await context.image_to_base64(self.start_image_url)
        arguments = {
            "output_format": self.output_format.value,
            "fps": self.fps,
            "sync_mode": self.sync_mode,
            "include_end": self.include_end,
            "include_start": self.include_start,
            "num_frames": self.num_frames,
            "end_image_url": f"data:image/png;base64,{end_image_url_base64}",
            "output_type": self.output_type.value,
            "start_image_url": f"data:image/png;base64,{start_image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/rife",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Film(FALNode):
    """
    FILM
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class VideoWriteMode(Enum):
        """
        The write mode of the output video. Only applicable if output_type is 'video'.
        """
        FAST = "fast"
        BALANCED = "balanced"
        SMALL = "small"

    class VideoQuality(Enum):
        """
        The quality of the output video. Only applicable if output_type is 'video'.
        """
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        MAXIMUM = "maximum"

    class ImageFormat(Enum):
        """
        The format of the output images. Only applicable if output_type is 'images'.
        """
        PNG = "png"
        JPEG = "jpeg"

    class OutputType(Enum):
        """
        The type of output to generate; either individual images or a video.
        """
        IMAGES = "images"
        VIDEO = "video"


    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the output video. Only applicable if output_type is 'video'."
    )
    num_frames: int = Field(
        default=1, description="The number of frames to generate between the input images."
    )
    include_start: bool = Field(
        default=False, description="Whether to include the start image in the output."
    )
    video_quality: VideoQuality = Field(
        default=VideoQuality.HIGH, description="The quality of the output video. Only applicable if output_type is 'video'."
    )
    include_end: bool = Field(
        default=False, description="Whether to include the end image in the output."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    fps: int = Field(
        default=8, description="Frames per second for the output video. Only applicable if output_type is 'video'."
    )
    start_image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the first image to use as the starting point for interpolation."
    )
    end_image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the second image to use as the ending point for interpolation."
    )
    image_format: ImageFormat = Field(
        default=ImageFormat.JPEG, description="The format of the output images. Only applicable if output_type is 'images'."
    )
    output_type: OutputType = Field(
        default=OutputType.IMAGES, description="The type of output to generate; either individual images or a video."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        start_image_url_base64 = await context.image_to_base64(self.start_image_url)
        end_image_url_base64 = await context.image_to_base64(self.end_image_url)
        arguments = {
            "video_write_mode": self.video_write_mode.value,
            "num_frames": self.num_frames,
            "include_start": self.include_start,
            "video_quality": self.video_quality.value,
            "include_end": self.include_end,
            "sync_mode": self.sync_mode,
            "fps": self.fps,
            "start_image_url": f"data:image/png;base64,{start_image_url_base64}",
            "end_image_url": f"data:image/png;base64,{end_image_url_base64}",
            "image_format": self.image_format.value,
            "output_type": self.output_type.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/film",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Calligrapher(FALNode):
    """
    Calligrapher
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    use_context: bool = Field(
        default=True, description="Whether to prepend context reference to the input"
    )
    num_images: int = Field(
        default=1, description="How many images to generate"
    )
    image_size: str = Field(
        default="", description="Target image size for generation"
    )
    auto_mask_generation: bool = Field(
        default=False, description="Whether to automatically generate mask from detected text"
    )
    reference_image_url: ImageRef = Field(
        default=ImageRef(), description="Optional base64 reference image for style"
    )
    source_image_url: ImageRef = Field(
        default=ImageRef(), description="Base64-encoded source image with drawn mask layers"
    )
    prompt: str = Field(
        default="", description="Text prompt to inpaint or customize"
    )
    mask_image_url: ImageRef = Field(
        default=ImageRef(), description="Base64-encoded mask image (optional if using auto_mask_generation)"
    )
    source_text: str = Field(
        default="", description="Source text to replace (if empty, masks all detected text)"
    )
    num_inference_steps: int = Field(
        default=50, description="Number of inference steps (1-100)"
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility"
    )
    cfg_scale: float = Field(
        default=1, description="Guidance or strength scale for the model"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        reference_image_url_base64 = await context.image_to_base64(self.reference_image_url)
        source_image_url_base64 = await context.image_to_base64(self.source_image_url)
        mask_image_url_base64 = await context.image_to_base64(self.mask_image_url)
        arguments = {
            "use_context": self.use_context,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "auto_mask_generation": self.auto_mask_generation,
            "reference_image_url": f"data:image/png;base64,{reference_image_url_base64}",
            "source_image_url": f"data:image/png;base64,{source_image_url_base64}",
            "prompt": self.prompt,
            "mask_image_url": f"data:image/png;base64,{mask_image_url_base64}",
            "source_text": self.source_text,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "cfg_scale": self.cfg_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/calligrapher",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class BriaReimagine(FALNode):
    """
    Bria
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    prompt: str = Field(
        default="", description="The prompt you would like to use to generate images."
    )
    num_results: int = Field(
        default=1, description="How many images you would like to generate. When using any Guidance Method, Value is set to 1."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    structure_ref_influence: float = Field(
        default=0.75, description="The influence of the structure reference on the generated image."
    )
    fast: bool = Field(
        default=True, description="Whether to use the fast model"
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    num_inference_steps: int = Field(
        default=30, description="The number of iterations the model goes through to refine the generated image. This parameter is optional."
    )
    structure_image_url: ImageRef = Field(
        default="", description="The URL of the structure reference image. Use \"\" to leave empty. Accepted formats are jpeg, jpg, png, webp."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        structure_image_url_base64 = await context.image_to_base64(self.structure_image_url)
        arguments = {
            "prompt": self.prompt,
            "num_results": self.num_results,
            "sync_mode": self.sync_mode,
            "structure_ref_influence": self.structure_ref_influence,
            "fast": self.fast,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "structure_image_url": f"data:image/png;base64,{structure_image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/bria/reimagine",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ImageEditingRealism(FALNode):
    """
    Image Editing
    editing, transformation, image-to-image, img2img

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    lora_scale: float = Field(
        default=0.6, description="The scale factor for the LoRA model. Controls the strength of the LoRA effect."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image to enhance with realism details."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_scale: float = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of inference steps for sampling."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "lora_scale": self.lora_scale,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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
            application="fal-ai/image-editing/realism",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PostProcessingVignette(FALNode):
    """
    Post Processing
    editing, transformation, image-to-image, img2img, professional

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    vignette_strength: float = Field(
        default=0.5, description="Vignette strength"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of image to process"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "vignette_strength": self.vignette_strength,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/post-processing/vignette",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PostProcessingSolarize(FALNode):
    """
    Post Processing
    editing, transformation, image-to-image, img2img, professional

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    solarize_threshold: float = Field(
        default=0.5, description="Solarize threshold"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of image to process"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "solarize_threshold": self.solarize_threshold,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/post-processing/solarize",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PostProcessingSharpen(FALNode):
    """
    Post Processing
    editing, transformation, image-to-image, img2img, professional

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class SharpenMode(Enum):
        """
        Type of sharpening to apply
        """
        BASIC = "basic"
        SMART = "smart"
        CAS = "cas"


    sharpen_mode: SharpenMode = Field(
        default=SharpenMode.BASIC, description="Type of sharpening to apply"
    )
    sharpen_alpha: float = Field(
        default=1, description="Sharpen strength (for basic mode)"
    )
    noise_radius: int = Field(
        default=7, description="Noise radius for smart sharpen"
    )
    sharpen_radius: int = Field(
        default=1, description="Sharpen radius (for basic mode)"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of image to process"
    )
    smart_sharpen_strength: float = Field(
        default=5, description="Smart sharpen strength"
    )
    cas_amount: float = Field(
        default=0.8, description="CAS sharpening amount"
    )
    preserve_edges: float = Field(
        default=0.75, description="Edge preservation factor"
    )
    smart_sharpen_ratio: float = Field(
        default=0.5, description="Smart sharpen blend ratio"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "sharpen_mode": self.sharpen_mode.value,
            "sharpen_alpha": self.sharpen_alpha,
            "noise_radius": self.noise_radius,
            "sharpen_radius": self.sharpen_radius,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "smart_sharpen_strength": self.smart_sharpen_strength,
            "cas_amount": self.cas_amount,
            "preserve_edges": self.preserve_edges,
            "smart_sharpen_ratio": self.smart_sharpen_ratio,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/post-processing/sharpen",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PostProcessingParabolize(FALNode):
    """
    Post Processing
    editing, transformation, image-to-image, img2img, professional

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    parabolize_coeff: float = Field(
        default=1, description="Parabolize coefficient"
    )
    vertex_y: float = Field(
        default=0.5, description="Vertex Y position"
    )
    vertex_x: float = Field(
        default=0.5, description="Vertex X position"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of image to process"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "parabolize_coeff": self.parabolize_coeff,
            "vertex_y": self.vertex_y,
            "vertex_x": self.vertex_x,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/post-processing/parabolize",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PostProcessingGrain(FALNode):
    """
    Post Processing
    editing, transformation, image-to-image, img2img, professional

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class GrainStyle(Enum):
        """
        Style of film grain to apply
        """
        MODERN = "modern"
        ANALOG = "analog"
        KODAK = "kodak"
        FUJI = "fuji"
        CINEMATIC = "cinematic"
        NEWSPAPER = "newspaper"


    grain_style: GrainStyle = Field(
        default=GrainStyle.MODERN, description="Style of film grain to apply"
    )
    grain_intensity: float = Field(
        default=0.4, description="Film grain intensity"
    )
    grain_scale: float = Field(
        default=10, description="Film grain scale"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of image to process"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "grain_style": self.grain_style.value,
            "grain_intensity": self.grain_intensity,
            "grain_scale": self.grain_scale,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/post-processing/grain",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PostProcessingDodgeBurn(FALNode):
    """
    Post Processing
    editing, transformation, image-to-image, img2img, professional

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class DodgeBurnMode(Enum):
        """
        Dodge and burn mode
        """
        DODGE = "dodge"
        BURN = "burn"
        DODGE_AND_BURN = "dodge_and_burn"
        BURN_AND_DODGE = "burn_and_dodge"
        COLOR_DODGE = "color_dodge"
        COLOR_BURN = "color_burn"
        LINEAR_DODGE = "linear_dodge"
        LINEAR_BURN = "linear_burn"


    dodge_burn_mode: DodgeBurnMode = Field(
        default=DodgeBurnMode.DODGE, description="Dodge and burn mode"
    )
    dodge_burn_intensity: float = Field(
        default=0.5, description="Dodge and burn intensity"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of image to process"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "dodge_burn_mode": self.dodge_burn_mode.value,
            "dodge_burn_intensity": self.dodge_burn_intensity,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/post-processing/dodge-burn",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PostProcessingDissolve(FALNode):
    """
    Post Processing
    editing, transformation, image-to-image, img2img, professional

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    dissolve_factor: float = Field(
        default=0.5, description="Dissolve blend factor"
    )
    dissolve_image_url: ImageRef = Field(
        default=ImageRef(), description="URL of second image for dissolve"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of image to process"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        dissolve_image_url_base64 = await context.image_to_base64(self.dissolve_image_url)
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "dissolve_factor": self.dissolve_factor,
            "dissolve_image_url": f"data:image/png;base64,{dissolve_image_url_base64}",
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/post-processing/dissolve",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PostProcessingDesaturate(FALNode):
    """
    Post Processing
    editing, transformation, image-to-image, img2img, professional

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class DesaturateMethod(Enum):
        """
        Desaturation method
        """
        LUMINANCE_REC_709 = "luminance (Rec.709)"
        LUMINANCE_REC_601 = "luminance (Rec.601)"
        AVERAGE = "average"
        LIGHTNESS = "lightness"


    desaturate_method: DesaturateMethod = Field(
        default=DesaturateMethod.LUMINANCE_REC_709, description="Desaturation method"
    )
    desaturate_factor: float = Field(
        default=1, description="Desaturation factor"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of image to process"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "desaturate_method": self.desaturate_method.value,
            "desaturate_factor": self.desaturate_factor,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/post-processing/desaturate",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PostProcessingColorTint(FALNode):
    """
    Post Processing
    editing, transformation, image-to-image, img2img, professional

    Use cases:
    - Professional photo editing and enhancement
    - Creative image transformations
    - Batch image processing workflows
    - Product photography refinement
    - Automated image optimization
    """

    class TintMode(Enum):
        """
        Tint color mode
        """
        SEPIA = "sepia"
        RED = "red"
        GREEN = "green"
        BLUE = "blue"
        CYAN = "cyan"
        MAGENTA = "magenta"
        YELLOW = "yellow"
        PURPLE = "purple"
        ORANGE = "orange"
        WARM = "warm"
        COOL = "cool"
        LIME = "lime"
        NAVY = "navy"
        VINTAGE = "vintage"
        ROSE = "rose"
        TEAL = "teal"
        MAROON = "maroon"
        PEACH = "peach"
        LAVENDER = "lavender"
        OLIVE = "olive"


    tint_strength: float = Field(
        default=1, description="Tint strength"
    )
    tint_mode: TintMode = Field(
        default=TintMode.SEPIA, description="Tint color mode"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of image to process"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "tint_strength": self.tint_strength,
            "tint_mode": self.tint_mode.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/post-processing/color-tint",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]