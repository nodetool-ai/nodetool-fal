from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.metadata.types import ImageRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class Ultrashape(FALNode):
    """
    Ultrashape
    3d_to_3d

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    octree_resolution: int = Field(
        default=1024, description="Marching cubes resolution."
    )
    remove_background: bool = Field(
        default=True, description="Remove image background."
    )
    num_inference_steps: int = Field(
        default=50, description="Diffusion steps."
    )
    model_url: str = Field(
        default="", description="URL of the coarse mesh (.glb or .obj) to refine."
    )
    seed: int = Field(
        default=42, description="Random seed."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the reference image for mesh refinement."
    )

    async def process(self, context: ProcessingContext) -> Any:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "octree_resolution": self.octree_resolution,
            "remove_background": self.remove_background,
            "num_inference_steps": self.num_inference_steps,
            "model_url": self.model_url,
            "seed": self.seed,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ultrashape",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["octree_resolution", "remove_background", "num_inference_steps", "model_url", "seed"]

class Sam33DAlign(FALNode):
    """
    Sam 3
    3d_to_3d

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the original image used for MoGe depth estimation"
    )
    body_mesh_url: str = Field(
        default="", description="URL of the SAM-3D Body mesh file (.ply or .glb) to align"
    )
    object_mesh_url: str = Field(
        default="", description="Optional URL of SAM-3D Object mesh (.glb) to create combined scene"
    )
    focal_length: float = Field(
        default=0.0, description="Focal length from SAM-3D Body metadata. If not provided, estimated from MoGe."
    )
    body_mask_url: ImageRef = Field(
        default=ImageRef(), description="URL of the human mask image. If not provided, uses full image."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_url_base64 = await context.image_to_base64(self.image_url)
        body_mask_url_base64 = await context.image_to_base64(self.body_mask_url)
        arguments = {
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "body_mesh_url": self.body_mesh_url,
            "object_mesh_url": self.object_mesh_url,
            "focal_length": self.focal_length,
            "body_mask_url": f"data:image/png;base64,{body_mask_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sam-3/3d-align",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["image_url", "body_mesh_url", "object_mesh_url", "focal_length", "body_mask_url"]