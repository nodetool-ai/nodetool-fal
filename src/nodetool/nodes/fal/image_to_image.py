from typing import Any
from pydantic import Field

from nodetool.metadata.types import ImageRef, VideoRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext
from .text_to_image import ImageSizePreset, HunyuanImageSizePreset


class FluxSchnellRedux(FALNode):
    """
    FLUX.1 [schnell] Redux is a high-performance endpoint that enables rapid transformation of existing images, delivering high-quality style transfers and image modifications with the core FLUX capabilities.
    image, transformation, style-transfer, fast, flux

    Use cases:
    - Transform images with style transfers
    - Apply artistic modifications to photos
    - Create image variations
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The input image to transform"
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3,
        description="The size of the generated image",
    )
    num_inference_steps: int = Field(
        default=4, ge=1, description="The number of inference steps to perform"
    )
    seed: int = Field(
        default=-1, description="The same seed will output the same image every time"
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux/schnell/redux",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "image_size", "num_inference_steps"]


class FluxDevRedux(FALNode):
    """
    FLUX.1 [dev] Redux is a high-performance endpoint that enables rapid transformation of existing images, delivering high-quality style transfers and image modifications.
    image, transformation, style-transfer, development, flux

    Use cases:
    - Transform images with advanced controls
    - Create customized image variations
    - Apply precise style modifications
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The input image to transform"
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3,
        description="The size of the generated image",
    )
    num_inference_steps: int = Field(
        default=28, ge=1, description="The number of inference steps to perform"
    )
    guidance_scale: float = Field(
        default=3.5, description="How closely the model should stick to your prompt"
    )
    seed: int = Field(
        default=-1, description="The same seed will output the same image every time"
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
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
            application="fal-ai/flux/dev/redux",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "image_size", "guidance_scale"]


class FluxProRedux(FALNode):
    """
    FLUX.1 [pro] Redux is a high-performance endpoint that enables rapid transformation of existing images, delivering high-quality style transfers and image modifications.
    image, transformation, style-transfer, flux

    Use cases:
    - Create professional image transformations
    - Generate style transfers
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The input image to transform"
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3,
        description="The size of the generated image",
    )
    num_inference_steps: int = Field(
        default=28, ge=1, description="The number of inference steps to perform"
    )
    guidance_scale: float = Field(
        default=3.5, description="How closely the model should stick to your prompt"
    )
    seed: int = Field(
        default=-1, description="The same seed will output the same image every time"
    )
    safety_tolerance: str = Field(
        default="2", description="Safety tolerance level (1-6, 1 being most strict)"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "safety_tolerance": self.safety_tolerance,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-pro/v1/redux",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "image_size", "guidance_scale"]


class FluxProUltraRedux(FALNode):
    """
    FLUX1.1 [pro] ultra Redux is a high-performance endpoint that enables rapid transformation of existing images, delivering high-quality style transfers and image modifications with the core FLUX capabilities.
    image, transformation, style-transfer, ultra, professional

    Use cases:
    - Apply precise image modifications
    - Process images with maximum control
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The input image to transform"
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3,
        description="The size of the generated image",
    )
    num_inference_steps: int = Field(
        default=28, ge=1, description="The number of inference steps to perform"
    )
    guidance_scale: float = Field(
        default=3.5, description="How closely the model should stick to your prompt"
    )
    seed: int = Field(
        default=-1, description="The same seed will output the same image every time"
    )
    safety_tolerance: str = Field(
        default="2", description="Safety tolerance level (1-6, 1 being most strict)"
    )
    image_prompt_strength: float = Field(
        default=0.1, description="The strength of the image prompt, between 0 and 1"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "safety_tolerance": self.safety_tolerance,
            "image_prompt_strength": self.image_prompt_strength,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-pro/v1.1-ultra/redux",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "image_size", "guidance_scale"]


class FluxProFill(FALNode):
    """FLUX.1 [pro] Fill is a high-performance endpoint that enables rapid transformation of existing images with inpainting/outpainting capabilities.
    image, inpainting, outpainting, transformation, professional

    Use cases:
    - Fill in missing or masked parts of images
    - Extend images beyond their original boundaries
    - Remove and replace unwanted elements
    - Create seamless image completions
    - Generate context-aware image content
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The input image to transform"
    )
    mask: ImageRef = Field(default=ImageRef(), description="The mask for inpainting")
    prompt: str = Field(
        default="", description="The prompt to fill the masked part of the image"
    )
    num_inference_steps: int = Field(
        default=28, ge=1, description="The number of inference steps to perform"
    )
    seed: int = Field(
        default=-1, description="The same seed will output the same image every time"
    )
    safety_tolerance: str = Field(
        default="2", description="Safety tolerance level (1-6, 1 being most strict)"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)
        mask_base64 = await context.image_to_base64(self.mask)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "mask_url": f"data:image/png;base64,{mask_base64}",
            "prompt": self.prompt,
            "num_inference_steps": self.num_inference_steps,
            "safety_tolerance": self.safety_tolerance,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-pro/v1/fill",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "mask", "prompt"]


class FluxProCanny(FALNode):
    """FLUX.1 [pro] Canny enables precise control over composition, style, and structure through advanced edge detection and guidance mechanisms.
    image, edge, composition, style, control

    Use cases:
    - Generate images with precise structural control
    - Create artwork based on edge maps
    - Transform sketches into detailed images
    - Maintain specific compositional elements
    - Generate variations with consistent structure
    """

    control_image: ImageRef = Field(
        default=ImageRef(),
        description="The control image to generate the Canny edge map from",
    )
    prompt: str = Field(default="", description="The prompt to generate an image from")
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3,
        description="The size of the generated image",
    )
    num_inference_steps: int = Field(
        default=28, ge=1, description="The number of inference steps to perform"
    )
    guidance_scale: float = Field(
        default=3.5, description="How closely the model should stick to your prompt"
    )
    seed: int = Field(
        default=-1, description="The same seed will output the same image every time"
    )
    safety_tolerance: str = Field(
        default="2", description="Safety tolerance level (1-6, 1 being most strict)"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        control_image_base64 = await context.image_to_base64(self.control_image)

        arguments = {
            "control_image_url": f"data:image/png;base64,{control_image_base64}",
            "prompt": self.prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "safety_tolerance": self.safety_tolerance,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-pro/v1/canny",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["control_image", "prompt", "image_size"]


class FluxProDepth(FALNode):
    """FLUX.1 [pro] Depth enables precise control over composition and structure through depth map detection and guidance mechanisms.
    image, depth-map, composition, structure, control

    Use cases:
    - Generate images with controlled depth perception
    - Create 3D-aware image transformations
    - Maintain spatial relationships in generated images
    - Produce images with accurate perspective
    - Generate variations with consistent depth structure
    """

    control_image: ImageRef = Field(
        default=ImageRef(),
        description="The control image to generate the depth map from",
    )
    prompt: str = Field(default="", description="The prompt to generate an image from")
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3,
        description="The size of the generated image",
    )
    num_inference_steps: int = Field(
        default=28, ge=1, description="The number of inference steps to perform"
    )
    guidance_scale: float = Field(
        default=3.5, description="How closely the model should stick to your prompt"
    )
    seed: int = Field(
        default=-1, description="The same seed will output the same image every time"
    )
    safety_tolerance: str = Field(
        default="2", description="Safety tolerance level (1-6, 1 being most strict)"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        control_image_base64 = await context.image_to_base64(self.control_image)

        arguments = {
            "control_image_url": f"data:image/png;base64,{control_image_base64}",
            "prompt": self.prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "safety_tolerance": self.safety_tolerance,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-pro/v1/depth",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["control_image", "prompt", "image_size"]


class FluxLoraCanny(FALNode):
    """FLUX LoRA Canny enables precise control over composition and style through edge detection and LoRA-based guidance mechanisms.
    image, edge, lora, style-transfer, control

    Use cases:
    - Generate stylized images with structural control
    - Create edge-guided artistic transformations
    - Apply custom styles while maintaining composition
    - Produce consistent style variations
    """

    control_image: ImageRef = Field(
        default=ImageRef(),
        description="The control image to generate the Canny edge map from",
    )
    prompt: str = Field(default="", description="The prompt to generate an image from")
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3,
        description="The size of the generated image",
    )
    num_inference_steps: int = Field(
        default=28, ge=1, description="The number of inference steps to perform"
    )
    guidance_scale: float = Field(
        default=3.5, description="How closely the model should stick to your prompt"
    )
    seed: int = Field(
        default=-1, description="The same seed will output the same image every time"
    )
    lora_scale: float = Field(
        default=0.6, description="The strength of the LoRA adaptation"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        control_image_base64 = await context.image_to_base64(self.control_image)

        arguments = {
            "control_image_url": f"data:image/png;base64,{control_image_base64}",
            "prompt": self.prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "lora_scale": self.lora_scale,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-lora-canny",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["control_image", "prompt", "image_size"]


class FluxLoraDepth(FALNode):
    """FLUX LoRA Depth enables precise control over composition and structure through depth map detection and LoRA-based guidance mechanisms.
    image, depth, lora, structure, control

    Use cases:
    - Generate depth-aware stylized images
    - Create 3D-conscious artistic transformations
    - Maintain spatial relationships with custom styles
    - Produce depth-consistent variations
    - Generate images with controlled perspective
    """

    control_image: ImageRef = Field(
        default=ImageRef(),
        description="The control image to generate the depth map from",
    )
    prompt: str = Field(default="", description="The prompt to generate an image from")
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3,
        description="The size of the generated image",
    )
    num_inference_steps: int = Field(
        default=28, ge=1, description="The number of inference steps to perform"
    )
    guidance_scale: float = Field(
        default=3.5, description="How closely the model should stick to your prompt"
    )
    seed: int = Field(
        default=-1, description="The same seed will output the same image every time"
    )
    lora_scale: float = Field(
        default=0.6, description="The strength of the LoRA adaptation"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        control_image_base64 = await context.image_to_base64(self.control_image)

        arguments = {
            "control_image_url": f"data:image/png;base64,{control_image_base64}",
            "prompt": self.prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "lora_scale": self.lora_scale,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-lora-depth",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["control_image", "prompt", "image_size"]


# ... existing code ...


class IdeogramV2Edit(FALNode):
    """Transform existing images with Ideogram V2's editing capabilities. Modify, adjust, and refine images while maintaining high fidelity and realistic outputs with precise prompt control.
    image, editing, transformation, fidelity, control

    Use cases:
    - Edit specific parts of images with precision
    - Create targeted image modifications
    - Refine and enhance image details
    - Generate contextual image edits
    """

    prompt: str = Field(
        default="", description="The prompt to fill the masked part of the image"
    )
    image: ImageRef = Field(default=ImageRef(), description="The image to edit")
    mask: ImageRef = Field(default=ImageRef(), description="The mask for editing")
    style: str = Field(
        default="auto",
        description="Style of generated image (auto, general, realistic, design, render_3D, anime)",
    )
    expand_prompt: bool = Field(
        default=True,
        description="Whether to expand the prompt with MagicPrompt functionality",
    )
    seed: int = Field(
        default=-1, description="The same seed will output the same image every time"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)
        mask_base64 = await context.image_to_base64(self.mask)

        arguments = {
            "prompt": self.prompt,
            "image_url": f"data:image/png;base64,{image_base64}",
            "mask_url": f"data:image/png;base64,{mask_base64}",
            "style": self.style,
            "expand_prompt": self.expand_prompt,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/ideogram/v2/edit",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image", "mask"]


class IdeogramV2Remix(FALNode):
    """Reimagine existing images with Ideogram V2's remix feature. Create variations and adaptations while preserving core elements and adding new creative directions through prompt guidance.
    image, remix, variation, creativity, adaptation

    Use cases:
    - Create artistic variations of images
    - Generate style-transferred versions
    - Produce creative image adaptations
    - Transform images while preserving key elements
    - Generate alternative interpretations
    """

    prompt: str = Field(default="", description="The prompt to remix the image with")
    image: ImageRef = Field(default=ImageRef(), description="The image to remix")
    aspect_ratio: str = Field(
        default="1:1", description="The aspect ratio of the generated image"
    )
    strength: float = Field(
        default=0.8, description="Strength of the input image in the remix"
    )
    expand_prompt: bool = Field(
        default=True,
        description="Whether to expand the prompt with MagicPrompt functionality",
    )
    style: str = Field(
        default="auto",
        description="Style of generated image (auto, general, realistic, design, render_3D, anime)",
    )
    seed: int = Field(
        default=-1, description="The same seed will output the same image every time"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "prompt": self.prompt,
            "image_url": f"data:image/png;base64,{image_base64}",
            "aspect_ratio": self.aspect_ratio,
            "strength": self.strength,
            "expand_prompt": self.expand_prompt,
            "style": self.style,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/ideogram/v2/remix",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image", "strength"]


class BriaEraser(FALNode):
    """
    Bria Eraser enables precise removal of unwanted objects from images while maintaining high-quality outputs. Trained exclusively on licensed data for safe and risk-free commercial use.
    image, removal, cleanup

    Use cases:
    - Remove unwanted objects from images
    - Clean up image imperfections
    - Prepare images for commercial use
    - Remove distracting elements
    - Create clean, professional images
    """

    image: ImageRef = Field(default=ImageRef(), description="Input image to erase from")
    mask: ImageRef = Field(
        default=ImageRef(), description="The mask for areas to be cleaned"
    )
    mask_type: str = Field(
        default="manual",
        description="Type of mask - 'manual' for user-created or 'automatic' for algorithm-generated",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)
        mask_base64 = await context.image_to_base64(self.mask)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "mask_url": f"data:image/png;base64,{mask_base64}",
            "mask_type": self.mask_type,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/bria/eraser",
            arguments=arguments,
        )
        assert "image" in res
        return ImageRef(uri=res["image"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "mask"]


class BriaProductShot(FALNode):
    """Place any product in any scenery with just a prompt or reference image while maintaining high integrity of the product. Trained exclusively on licensed data for safe and risk-free commercial use and optimized for eCommerce.
    image, product, placement, ecommerce

    Use cases:
    - Create professional product photography
    - Generate contextual product shots
    - Place products in custom environments
    - Create eCommerce product listings
    - Generate marketing visuals
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The product image to be placed"
    )
    scene_description: str = Field(
        default="", description="Text description of the new scene/background"
    )
    ref_image: ImageRef = Field(
        default=ImageRef(), description="Reference image for the new scene/background"
    )
    optimize_description: bool = Field(
        default=True, description="Whether to optimize the scene description"
    )
    placement_type: str = Field(
        default="manual_placement",
        description="How to position the product (original, automatic, manual_placement, manual_padding)",
    )
    manual_placement_selection: str = Field(
        default="bottom_center",
        description="Specific placement position when using manual_placement",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)
        ref_image_base64 = (
            await context.image_to_base64(self.ref_image) if self.ref_image.uri else ""
        )

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "scene_description": self.scene_description,
            "ref_image_url": (
                f"data:image/png;base64,{ref_image_base64}" if ref_image_base64 else ""
            ),
            "optimize_description": self.optimize_description,
            "placement_type": self.placement_type,
            "manual_placement_selection": self.manual_placement_selection,
            "shot_size": [1000, 1000],
            "fast": True,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/bria/product-shot",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "scene_description"]


class BriaBackgroundReplace(FALNode):
    """Bria Background Replace allows for efficient swapping of backgrounds in images via text prompts or reference image, delivering realistic and polished results. Trained exclusively on licensed data for safe and risk-free commercial use.
    image, background, replacement, swap

    Use cases:
    - Replace image backgrounds seamlessly
    - Create professional photo compositions
    - Generate custom scene settings
    - Produce commercial-ready images
    - Create consistent visual environments
    """

    image: ImageRef = Field(
        default=ImageRef(), description="Input image to replace background"
    )
    ref_image: ImageRef = Field(
        default=ImageRef(), description="Reference image for the new background"
    )
    prompt: str = Field(default="", description="Prompt to generate new background")
    negative_prompt: str = Field(
        default="", description="Negative prompt for background generation"
    )
    refine_prompt: bool = Field(
        default=True, description="Whether to refine the prompt"
    )
    seed: int = Field(
        default=-1, description="The same seed will output the same image every time"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)
        ref_image_base64 = (
            await context.image_to_base64(self.ref_image) if self.ref_image.uri else ""
        )

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "ref_image_url": (
                f"data:image/png;base64,{ref_image_base64}" if ref_image_base64 else ""
            ),
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "refine_prompt": self.refine_prompt,
            "fast": True,
            "num_images": 1,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/bria/background/replace",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]


class BriaGenFill(FALNode):
    """Bria GenFill enables high-quality object addition or visual transformation. Trained exclusively on licensed data for safe and risk-free commercial use.
    image, generation, filling, transformation

    Use cases:
    - Add new objects to existing images
    - Transform specific image areas
    - Generate contextual content
    - Create seamless visual additions
    - Produce professional image modifications
    """

    image: ImageRef = Field(default=ImageRef(), description="Input image to erase from")
    mask: ImageRef = Field(
        default=ImageRef(), description="The mask for areas to be cleaned"
    )
    prompt: str = Field(default="", description="The prompt to generate images")
    negative_prompt: str = Field(
        default="", description="The negative prompt to use when generating images"
    )
    seed: int = Field(
        default=-1, description="The same seed will output the same image every time"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)
        mask_base64 = await context.image_to_base64(self.mask)

        arguments: dict[str, Any] = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "mask_url": f"data:image/png;base64,{mask_base64}",
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/bria/genfill",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "mask", "prompt"]


class BriaExpand(FALNode):
    """Bria Expand expands images beyond their borders in high quality. Trained exclusively on licensed data for safe and risk-free commercial use.
    image, expansion, outpainting

    Use cases:
    - Extend image boundaries seamlessly
    - Create wider or taller compositions
    - Expand images for different aspect ratios
    - Generate additional scene content
    """

    image: ImageRef = Field(default=ImageRef(), description="The input image to expand")
    canvas_width: int = Field(
        default=1200,
        description="The desired width of the final image, after the expansion",
    )
    canvas_height: int = Field(
        default=674,
        description="The desired height of the final image, after the expansion",
    )
    original_image_width: int = Field(
        default=610,
        description="The desired width of the original image, inside the full canvas",
    )
    original_image_height: int = Field(
        default=855,
        description="The desired height of the original image, inside the full canvas",
    )
    original_image_x: int = Field(
        default=301,
        description="The desired x-coordinate of the original image, inside the full canvas",
    )
    original_image_y: int = Field(
        default=-66,
        description="The desired y-coordinate of the original image, inside the full canvas",
    )
    prompt: str = Field(
        default="", description="Text on which you wish to base the image expansion"
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use when generating images"
    )
    num_images: int = Field(default=1, description="Number of images to generate")
    seed: int = Field(
        default=-1, description="The same seed will output the same image every time"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "canvas_size": [self.canvas_width, self.canvas_height],
            "original_image_size": [
                self.original_image_width,
                self.original_image_height,
            ],
            "original_image_location": [self.original_image_x, self.original_image_y],
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "num_images": self.num_images,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/bria/expand",
            arguments=arguments,
        )
        assert "image" in res
        return ImageRef(uri=res["image"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "canvas_width", "canvas_height", "prompt"]


class BriaBackgroundRemove(FALNode):
    """
    Bria RMBG 2.0 enables seamless removal of backgrounds from images, ideal for professional editing tasks.
    Trained exclusively on licensed data for safe and risk-free commercial use.
    """

    image: ImageRef = Field(
        default=ImageRef(), description="Input image to remove background from"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {"image_url": f"data:image/png;base64,{image_base64}"}

        res = await self.submit_request(
            context=context,
            application="fal-ai/bria/background/remove",
            arguments=arguments,
        )
        assert "image" in res
        return ImageRef(uri=res["image"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image"]


class ClarityUpscaler(FALNode):
    """Upscale images to improve resolution and sharpness.

    clarity, upscale, enhancement

    Use cases:
    - Increase image resolution for printing
    - Improve clarity of low-quality images
    - Enhance textures and graphics
    """

    image: ImageRef = Field(
        default=ImageRef(),
        description="Input image to upscale",
    )
    scale: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Upscaling factor",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "scale": self.scale,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/clarity-upscaler",
            arguments=arguments,
        )
        if "image" in res:
            return ImageRef(uri=res["image"]["url"])
        assert res.get("images") is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "scale"]


class WanEffects(FALNode):
    """Apply stylized effects to an image using the WAN Effects model.

    image, transformation, style, filter

    Use cases:
    - Add artistic filters to photos
    - Create stylized social media images
    - Quickly generate meme-style effects
    """

    image: ImageRef = Field(
        default=ImageRef(),
        description="Input image to apply the effect to",
    )
    effect: str = Field(default="", description="Name of the effect to apply")

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "effect": self.effect,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-effects",
            arguments=arguments,
        )
        if "image" in res:
            return ImageRef(uri=res["image"]["url"])
        assert res.get("images") is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "effect"]


class IdeogramV3Edit(FALNode):
    """
    Ideogram V3 Edit for editing images with text prompts while maintaining structure.
    image, editing, ideogram, inpainting, text-guided

    Use cases:
    - Edit specific parts of images with prompts
    - Modify text in images
    - Change elements while preserving composition
    - Add or remove objects from images
    - Refine generated images
    """

    image: ImageRef = Field(default=ImageRef(), description="The input image to edit")
    mask: ImageRef = Field(
        default=ImageRef(), description="The mask indicating areas to edit"
    )
    prompt: str = Field(default="", description="The prompt describing the edit")
    style: str = Field(default="auto", description="The style of the edit")
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)
        mask_base64 = await context.image_to_base64(self.mask)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "mask_url": f"data:image/png;base64,{mask_base64}",
            "prompt": self.prompt,
            "style": self.style,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/ideogram/v3/edit",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "mask", "prompt"]


class GPTImage1Edit(FALNode):
    """
    OpenAI GPT Image 1 Edit for modifying images with text instructions.
    image, editing, openai, gpt, text-guided

    Use cases:
    - Edit images with natural language
    - Modify specific elements in photos
    - Add or change objects
    - Apply creative edits
    - Refine images iteratively
    """

    image: ImageRef = Field(default=ImageRef(), description="The input image to edit")
    mask: ImageRef = Field(
        default=ImageRef(), description="The mask for inpainting (optional)"
    )
    prompt: str = Field(default="", description="Instructions for editing the image")
    size: str = Field(default="1024x1024", description="The size of the output image")
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
            "size": self.size,
        }
        if self.mask and self.mask.uri:
            mask_base64 = await context.image_to_base64(self.mask)
            arguments["mask_url"] = f"data:image/png;base64,{mask_base64}"
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/gpt-image-1/edit-image",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]


class GeminiFlashEdit(FALNode):
    """
    Google Gemini Flash Edit for fast image editing with text prompts.
    image, editing, google, gemini, fast, text-guided

    Use cases:
    - Quick image modifications
    - Fast iterative edits
    - Object addition or removal
    - Style adjustments
    - Rapid prototyping
    """

    image: ImageRef = Field(default=ImageRef(), description="The input image to edit")
    prompt: str = Field(default="", description="Instructions for editing the image")
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/gemini-flash-edit",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]


class Flux2TurboEdit(FALNode):
    """
    FLUX 2 Turbo Edit for fast image editing with the FLUX 2 model.
    image, editing, flux, fast, turbo, text-guided

    Use cases:
    - Rapid image modifications
    - Quick style transfers
    - Fast object editing
    - Iterative refinement
    - Real-time editing workflows
    """

    image: ImageRef = Field(default=ImageRef(), description="The input image to edit")
    prompt: str = Field(default="", description="The prompt describing the edit")
    num_inference_steps: int = Field(
        default=4, ge=1, description="Number of inference steps"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
            "num_inference_steps": self.num_inference_steps,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2/turbo/edit",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]


class CreativeUpscaler(FALNode):
    """
    Creative Upscaler enhances image resolution while adding creative details.
    image, upscaling, enhancement, super-resolution, creative

    Use cases:
    - Upscale low-resolution images
    - Enhance image details creatively
    - Improve image quality
    - Prepare images for print
    - Restore old or compressed images
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The input image to upscale"
    )
    prompt: str = Field(
        default="", description="Optional prompt to guide the upscaling"
    )
    scale: float = Field(default=2.0, ge=1.0, le=4.0, description="Upscaling factor")
    creativity: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Level of creative enhancement"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "scale": self.scale,
            "creativity": self.creativity,
        }
        if self.prompt:
            arguments["prompt"] = self.prompt
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/creative-upscaler",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "scale", "creativity"]


class BiRefNet(FALNode):
    """
    BiRefNet is a high-quality background removal model using bilateral reference.
    image, background-removal, segmentation, matting

    Use cases:
    - Remove backgrounds from photos
    - Create product images with transparent backgrounds
    - Extract subjects from images
    - Prepare images for compositing
    - Create stickers and cutouts
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The input image for background removal"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/birefnet",
            arguments=arguments,
        )
        assert res["image"] is not None
        return ImageRef(uri=res["image"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image"]


class BiRefNetV2(FALNode):
    """
    BiRefNet V2 is an improved background removal model with better edge detection.
    image, background-removal, segmentation, matting, v2

    Use cases:
    - High-quality background removal
    - Precise edge detection for cutouts
    - Product photography processing
    - Portrait extraction
    - Complex background handling
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The input image for background removal"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/birefnet/v2",
            arguments=arguments,
        )
        assert res["image"] is not None
        return ImageRef(uri=res["image"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image"]


class CodeFormer(FALNode):
    """
    CodeFormer is a face restoration model for enhancing and restoring face quality.
    image, face-restoration, enhancement, quality

    Use cases:
    - Restore old or damaged photos
    - Enhance low-quality face images
    - Improve portrait quality
    - Fix facial artifacts
    - Upscale face details
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The input image with faces to restore"
    )
    fidelity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Balance between quality and fidelity (0=quality, 1=fidelity)",
    )
    background_enhance: bool = Field(
        default=True, description="Whether to enhance the background"
    )
    face_upsample: bool = Field(
        default=True, description="Whether to upsample the face"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "fidelity": self.fidelity,
            "background_enhance": self.background_enhance,
            "face_upsample": self.face_upsample,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/codeformer",
            arguments=arguments,
        )
        assert res["image"] is not None
        return ImageRef(uri=res["image"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "fidelity"]


class ESRGAN(FALNode):
    """
    ESRGAN (Enhanced Super-Resolution GAN) for high-quality image upscaling.
    image, upscaling, super-resolution, enhancement

    Use cases:
    - Upscale images to higher resolution
    - Enhance image details
    - Improve image quality for printing
    - Restore low-resolution images
    - Prepare images for large displays
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The input image to upscale"
    )
    scale: int = Field(default=4, ge=2, le=8, description="Upscaling factor")

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "scale": self.scale,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/esrgan",
            arguments=arguments,
        )
        assert res["image"] is not None
        return ImageRef(uri=res["image"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "scale"]


class ImageUtilsRembg(FALNode):
    """
    Rembg utility for removing image backgrounds with high accuracy.
    image, background-removal, utility, processing

    Use cases:
    - Remove backgrounds from product photos
    - Create transparent PNG images
    - Extract subjects for compositing
    - Prepare images for design work
    - Create profile picture cutouts
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The input image for background removal"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/imageutils/rembg",
            arguments=arguments,
        )
        assert res["image"] is not None
        return ImageRef(uri=res["image"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image"]


class ImageUtilsDepth(FALNode):
    """
    Depth estimation utility for generating depth maps from images.
    image, depth-map, estimation, 3d, utility

    Use cases:
    - Generate depth maps for 3D effects
    - Create parallax animations
    - Enable depth-aware editing
    - Generate ControlNet inputs
    - Analyze image depth structure
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The input image for depth estimation"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/imageutils/depth",
            arguments=arguments,
        )
        assert res["image"] is not None
        return ImageRef(uri=res["image"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image"]


class LivePortrait(FALNode):
    """
    Live Portrait animates a single portrait image based on a driving video.
    image, animation, portrait, face, motion-transfer

    Use cases:
    - Animate static portraits
    - Create talking head videos
    - Transfer facial expressions
    - Create avatar animations
    - Generate video from single photo
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The portrait image to animate"
    )
    driving_video: VideoRef = Field(
        default=VideoRef(), description="The driving video with motion reference"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        from nodetool.metadata.types import VideoRef as VideoRefType

        client = await self.get_client(context)
        image_base64 = await context.image_to_base64(self.image)
        video_bytes = await context.asset_to_bytes(self.driving_video)
        video_url = await client.upload(video_bytes, "video/mp4")

        arguments = {
            "source_image_url": f"data:image/png;base64,{image_base64}",
            "driving_video_url": video_url,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/live-portrait",
            arguments=arguments,
        )
        assert res["video"] is not None
        return VideoRefType(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "driving_video"]


class PuLID(FALNode):
    """
    PuLID generates images with consistent face identity from a reference face.
    image, face, identity, generation, consistency

    Use cases:
    - Generate images with consistent face identity
    - Create character variations
    - Design personalized avatars
    - Produce face-consistent content
    - Generate marketing images with specific faces
    """

    face_image: ImageRef = Field(
        default=ImageRef(), description="The reference face image"
    )
    prompt: str = Field(
        default="", description="The prompt describing the desired image"
    )
    negative_prompt: str = Field(
        default="", description="What to avoid in the generated image"
    )
    num_inference_steps: int = Field(
        default=20, ge=1, description="Number of inference steps"
    )
    guidance_scale: float = Field(
        default=1.2, description="How closely to follow the prompt"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> ImageRef:
        face_base64 = await context.image_to_base64(self.face_image)

        arguments = {
            "reference_images": [f"data:image/png;base64,{face_base64}"],
            "prompt": self.prompt,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }
        if self.negative_prompt:
            arguments["negative_prompt"] = self.negative_prompt
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/pulid",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["face_image", "prompt"]


class PhotoMaker(FALNode):
    """
    PhotoMaker generates images with customizable subject identity from reference photos.
    image, face, identity, customization, generation

    Use cases:
    - Generate images with specific person identity
    - Create personalized marketing content
    - Design custom avatars
    - Produce character-consistent images
    - Generate variations of a person
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The reference image of the subject"
    )
    prompt: str = Field(
        default="", description="The prompt describing the desired image"
    )
    negative_prompt: str = Field(
        default="", description="What to avoid in the generated image"
    )
    num_inference_steps: int = Field(
        default=50, ge=1, description="Number of inference steps"
    )
    style_strength: float = Field(
        default=20.0, description="Strength of the style transfer"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_archive_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
            "num_inference_steps": self.num_inference_steps,
            "style_strength": self.style_strength,
        }
        if self.negative_prompt:
            arguments["negative_prompt"] = self.negative_prompt
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/photomaker",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]


class FaceToSticker(FALNode):
    """
    Face to Sticker transforms face photos into fun sticker-style images.
    image, face, sticker, fun, transformation

    Use cases:
    - Create fun stickers from photos
    - Generate emoji-style faces
    - Design personalized sticker packs
    - Create cartoon avatars
    - Produce fun social media content
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The face image to convert to sticker"
    )
    prompt: str = Field(
        default="sticker", description="Optional prompt to guide the sticker style"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
        }
        if self.prompt:
            arguments["prompt"] = self.prompt
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/face-to-sticker",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image"]


class Cartoonify(FALNode):
    """
    Cartoonify transforms photos into cartoon-style images.
    image, cartoon, style-transfer, fun, artistic

    Use cases:
    - Convert photos to cartoon style
    - Create animated-style portraits
    - Design fun profile pictures
    - Generate cartoon avatars
    - Create artistic transformations
    """

    image: ImageRef = Field(default=ImageRef(), description="The image to cartoonify")

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/cartoonify",
            arguments=arguments,
        )
        assert res["image"] is not None
        return ImageRef(uri=res["image"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image"]


class KolorsImageToImage(FALNode):
    """
    Kolors Image-to-Image transforms images with the Kolors model.
    image, transformation, kolors, style-transfer

    Use cases:
    - Transform image style
    - Apply artistic effects
    - Modify image content
    - Create style variations
    - Generate image edits
    """

    image: ImageRef = Field(default=ImageRef(), description="The input image")
    prompt: str = Field(
        default="", description="The prompt describing the transformation"
    )
    negative_prompt: str = Field(
        default="", description="What to avoid in the generated image"
    )
    strength: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Transformation strength"
    )
    num_inference_steps: int = Field(
        default=25, ge=1, description="Number of inference steps"
    )
    guidance_scale: float = Field(
        default=5.0, description="How closely to follow the prompt"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
            "strength": self.strength,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }
        if self.negative_prompt:
            arguments["negative_prompt"] = self.negative_prompt
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/kolors/image-to-image",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt", "strength"]


class ObjectRemoval(FALNode):
    """
    Object Removal removes unwanted objects from images using AI.
    image, inpainting, removal, cleanup

    Use cases:
    - Remove unwanted objects from photos
    - Clean up image backgrounds
    - Remove watermarks or logos
    - Fix photo imperfections
    - Create clean product shots
    """

    image: ImageRef = Field(default=ImageRef(), description="The input image")
    mask: ImageRef = Field(
        default=ImageRef(), description="Mask indicating objects to remove"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)
        mask_base64 = await context.image_to_base64(self.mask)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "mask_url": f"data:image/png;base64,{mask_base64}",
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/object-removal",
            arguments=arguments,
        )
        assert res["image"] is not None
        return ImageRef(uri=res["image"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "mask"]


class Retoucher(FALNode):
    """
    Retoucher enhances and retouches photos with AI-powered corrections.
    image, enhancement, retouching, beautification

    Use cases:
    - Enhance portrait photos
    - Apply skin retouching
    - Improve photo quality
    - Fix lighting issues
    - Professional photo editing
    """

    image: ImageRef = Field(default=ImageRef(), description="The image to retouch")
    prompt: str = Field(
        default="", description="Optional prompt to guide the retouching"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
        }
        if self.prompt:
            arguments["prompt"] = self.prompt

        res = await self.submit_request(
            context=context,
            application="fal-ai/retoucher",
            arguments=arguments,
        )
        assert res["image"] is not None
        return ImageRef(uri=res["image"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image"]


class CCSR(FALNode):
    """
    CCSR (Content-Consistent Super-Resolution) for high-quality image upscaling.
    image, upscaling, super-resolution, enhancement

    Use cases:
    - Upscale images with content consistency
    - Enhance low-resolution photos
    - Improve image details
    - Prepare images for printing
    - Restore compressed images
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The input image to upscale"
    )
    scale: int = Field(default=4, ge=2, le=4, description="Upscaling factor")

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "scale": self.scale,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/ccsr",
            arguments=arguments,
        )
        assert res["image"] is not None
        return ImageRef(uri=res["image"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "scale"]


class HunyuanImageV3InstructEdit(FALNode):
    """
    Hunyuan Image V3 Instruct Edit with reasoning capabilities for advanced image-to-image editing.
    image, edit, hunyuan, tencent, instruct, reasoning, image-to-image, img2img, advanced

    Use cases:
    - Edit images with complex instructions
    - Apply style transfers with reasoning
    - Modify images with multiple reference images
    - Create variations with intelligent understanding
    - Transform images with advanced prompt interpretation
    """

    prompt: str = Field(default="", description="The text prompt for editing the image")
    image_urls: list[ImageRef] = Field(
        default_factory=list,
        description="Reference images to use (maximum 2 images)",
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
        # Convert images to base64
        image_urls_base64 = []
        for image_ref in self.image_urls[:2]:  # Maximum 2 images
            if image_ref.uri:
                image_base64 = await context.image_to_base64(image_ref)
                image_urls_base64.append(f"data:image/png;base64,{image_base64}")

        arguments = {
            "prompt": self.prompt,
            "image_urls": image_urls_base64,
            "image_size": self.image_size.value,
            "num_images": self.num_images,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan-image/v3/instruct/edit",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_urls", "guidance_scale"]


class QwenImageMaxEdit(FALNode):
    """
    Qwen Image Max Edit for advanced image editing with reference images.
    image, edit, qwen, alibaba, image-to-image, img2img, high-quality

    Use cases:
    - Edit images with complex instructions
    - Transform images based on references
    - Apply style transfers with multiple images
    - Create variations with intelligent editing
    - Modify images with detailed prompts
    """

    prompt: str = Field(
        default="", description="Text prompt describing the desired edits"
    )
    negative_prompt: str = Field(
        default="", description="Content to avoid in the edited image"
    )
    image_urls: list[ImageRef] = Field(
        default_factory=list,
        description="Reference images for editing (1-3 images)",
    )
    image_size: ImageSizePreset | None = Field(
        default=None,
        description="The size of the generated image. If not provided, uses input image size",
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Enable LLM prompt optimization for better results"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")
    enable_safety_checker: bool = Field(
        default=True, description="Enable content moderation"
    )
    num_images: int = Field(
        default=1, ge=1, le=4, description="The number of images to generate"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        # Convert images to base64
        image_urls_base64 = []
        for image_ref in self.image_urls[:3]:  # Maximum 3 images
            if image_ref.uri:
                image_base64 = await context.image_to_base64(image_ref)
                image_urls_base64.append(f"data:image/png;base64,{image_base64}")

        arguments = {
            "prompt": self.prompt,
            "image_urls": image_urls_base64,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "enable_safety_checker": self.enable_safety_checker,
            "num_images": self.num_images,
        }
        if self.negative_prompt:
            arguments["negative_prompt"] = self.negative_prompt
        if self.image_size:
            arguments["image_size"] = self.image_size.value
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-max/edit",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_urls", "enable_prompt_expansion"]


class BriaReplaceBackground(FALNode):
    """
    Creates enriched product shots by placing them in various environments using textual descriptions.
    image, background, replacement, product, enhancement, bria

    Use cases:
    - Replace product image backgrounds with custom environments
    - Create professional product photography
    - Generate contextual product shots
    - Enhance e-commerce product images
    - Create marketing visuals with custom backgrounds
    """

    image: ImageRef = Field(
        default=ImageRef(), description="Reference image for background replacement"
    )
    prompt: str = Field(
        default="", description="Prompt for background replacement"
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt for background replacement"
    )
    seed: int = Field(
        default=4925634, description="Random seed for reproducibility"
    )
    steps_num: int = Field(
        default=30, description="Number of inference steps"
    )
    sync_mode: bool = Field(
        default=False,
        description="If true, returns the image directly in the response (increases latency)",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "prompt": self.prompt,
            "image_url": f"data:image/png;base64,{image_base64}",
            "seed": self.seed,
            "steps_num": self.steps_num,
        }
        if self.negative_prompt:
            arguments["negative_prompt"] = self.negative_prompt
        if self.sync_mode:
            arguments["sync_mode"] = self.sync_mode

        res = await self.submit_request(
            context=context,
            application="bria/replace-background",
            arguments=arguments,
        )
        assert "image" in res
        return ImageRef(uri=res["image"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt", "steps_num"]


class FaceSwapImage(FALNode):
    """
    Swap faces between source and target images. Creates realistic face swaps with optional occlusion prevention for handling objects covering faces.
    face-swap, face-transfer, image-manipulation, face-replacement, portrait

    Use cases:
    - Swap faces in photos for creative content
    - Create fun photo edits with friend's faces
    - Generate alternative portraits
    - Test how you'd look with different hairstyles
    - Create face-swapped memes and social content
    """

    source_face: ImageRef = Field(
        default=ImageRef(), description="Source face image to swap from"
    )
    target_image: ImageRef = Field(
        default=ImageRef(), description="Target image to swap face into"
    )
    enable_occlusion_prevention: bool = Field(
        default=False,
        description="Enable occlusion prevention for faces covered by hands/objects (costs 2x more)",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        source_base64 = await context.image_to_base64(self.source_face)
        target_base64 = await context.image_to_base64(self.target_image)

        arguments = {
            "source_face_url": f"data:image/png;base64,{source_base64}",
            "target_image_url": f"data:image/png;base64,{target_base64}",
        }

        if self.enable_occlusion_prevention:
            arguments["enable_occlusion_prevention"] = self.enable_occlusion_prevention

        res = await self.submit_request(
            context=context,
            application="half-moon-ai/ai-face-swap/faceswapimage",
            arguments=arguments,
        )
        assert "image" in res
        return ImageRef(uri=res["image"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["source_face", "target_image"]
