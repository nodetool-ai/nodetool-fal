from typing import TypedDict
from enum import Enum
from pydantic import Field

from nodetool.metadata.types import ImageRef, Model3DRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class TextureSizeEnum(Enum):
    SIZE_512 = 512
    SIZE_1024 = 1024
    SIZE_2048 = 2048


class Trellis(FALNode):
    """
    Trellis generates 3D models from single images.
    3d, generation, image-to-3d, trellis

    Use cases:
    - Generate 3D models from images
    - Create 3D assets from photos
    - Produce 3D content for games
    - Create 3D visualizations
    - Generate 3D for AR/VR
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to convert to 3D"
    )
    ss_guidance_strength: float = Field(
        default=7.5, ge=0.0, le=20.0, description="Guidance strength for sparse structure"
    )
    ss_sampling_steps: int = Field(
        default=12, ge=1, le=50, description="Sampling steps for sparse structure"
    )
    slat_guidance_strength: float = Field(
        default=3.0, ge=0.0, le=20.0, description="Guidance strength for structured latent"
    )
    slat_sampling_steps: int = Field(
        default=12, ge=1, le=50, description="Sampling steps for structured latent"
    )
    mesh_simplify: float = Field(
        default=0.95, ge=0.9, le=0.98, description="Mesh simplification ratio"
    )
    texture_size: TextureSizeEnum = Field(
        default=TextureSizeEnum.SIZE_1024, description="Texture resolution"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> Model3DRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "ss_guidance_strength": self.ss_guidance_strength,
            "ss_sampling_steps": self.ss_sampling_steps,
            "slat_guidance_strength": self.slat_guidance_strength,
            "slat_sampling_steps": self.slat_sampling_steps,
            "mesh_simplify": self.mesh_simplify,
            "texture_size": self.texture_size.value,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/trellis",
            arguments=arguments,
        )
        return Model3DRef(uri=res.get("glb", {}).get("url", ""))

    @classmethod
    def get_basic_fields(cls):
        return ["image", "texture_size"]


class Hunyuan3DV2(FALNode):
    """
    Hunyuan3D V2 generates high-quality 3D models from images.
    3d, generation, image-to-3d, hunyuan

    Use cases:
    - Generate detailed 3D models
    - Create 3D assets from photos
    - Produce high-quality 3D content
    - Create 3D visualizations
    - Generate 3D for productions
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to convert to 3D"
    )
    num_inference_steps: int = Field(
        default=50, ge=1, le=100, description="Number of inference steps"
    )
    guidance_scale: float = Field(
        default=2.0, ge=1.0, le=10.0, description="Guidance scale for generation"
    )
    octree_resolution: int = Field(
        default=256, description="Octree resolution for 3D structure"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> Model3DRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "input_image_url": f"data:image/png;base64,{image_base64}",
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "octree_resolution": self.octree_resolution,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan3d/v2",
            arguments=arguments,
        )
        return Model3DRef(uri=res.get("model_mesh", {}).get("url", ""))

    @classmethod
    def get_basic_fields(cls):
        return ["image"]


class TripoSR(FALNode):
    """
    TripoSR generates 3D models from images with fast processing.
    3d, generation, image-to-3d, triposr, fast

    Use cases:
    - Quick 3D model generation
    - Rapid prototyping
    - Create 3D assets from photos
    - Generate 3D content quickly
    - Fast 3D for AR/VR
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The input image to convert to 3D"
    )
    foreground_ratio: float = Field(
        default=0.85, ge=0.5, le=1.0, description="Foreground ratio for cropping"
    )

    async def process(self, context: ProcessingContext) -> Model3DRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "foreground_ratio": self.foreground_ratio,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/triposr",
            arguments=arguments,
        )
        return Model3DRef(uri=res.get("model", {}).get("url", ""))

    @classmethod
    def get_basic_fields(cls):
        return ["image"]

    @classmethod
    def return_type(cls):
        return Model3DRef


class Era3D(FALNode):
    """
    Era3D creates multi-view consistent 3D models from images.
    3d, generation, image-to-3d, era3d, multi-view

    Use cases:
    - Generate multi-view 3D models
    - Create consistent 3D assets
    - Produce 3D content with multiple views
    - Generate detailed 3D models
    - Create multi-view 3D for games
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to convert to 3D"
    )
    num_inference_steps: int = Field(
        default=40, ge=10, le=100, description="Number of inference steps"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    class OutputType(TypedDict):
        mv_images: list
        model: Model3DRef

    async def process(self, context: ProcessingContext) -> OutputType:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "num_inference_steps": self.num_inference_steps,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/era-3d",
            arguments=arguments,
        )
        return {
            "mv_images": [ImageRef(uri=img["url"]) for img in res.get("mv_images", [])],
            "model": Model3DRef(uri=res.get("model", {}).get("url", "")),
        }

    @classmethod
    def get_basic_fields(cls):
        return ["image"]