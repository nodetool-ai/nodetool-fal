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

class MeshyV5Retexture(FALNode):
    """
    Meshy-5 retexture applies new, high-quality textures to existing 3D models using either text prompts or reference images. It supports PBR material generation for realistic, production-ready results.
    3d, editing, transformation, modeling

    Use cases:
    - 3D model editing and refinement
    - Mesh optimization
    - Texture application
    - 3D format conversion
    - Model retopology
    """

    enable_pbr: bool = Field(
        default=False, description="Generate PBR Maps (metallic, roughness, normal) in addition to base color."
    )
    text_style_prompt: str = Field(
        default="", description="Describe your desired texture style using text. Maximum 600 characters. Required if image_style_url is not provided."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, input data will be checked for safety before processing."
    )
    enable_original_uv: bool = Field(
        default=True, description="Use the original UV mapping of the model instead of generating new UVs. If the model has no original UV, output quality may be reduced."
    )
    model_url: str = Field(
        default="", description="URL or base64 data URI of a 3D model to texture. Supports .glb, .gltf, .obj, .fbx, .stl formats. Can be a publicly accessible URL or data URI with MIME type application/octet-stream."
    )
    image_style_url: ImageRef = Field(
        default=ImageRef(), description="2D image to guide the texturing process. Supports .jpg, .jpeg, and .png formats. Required if text_style_prompt is not provided. If both are provided, image_style_url takes precedence."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_style_url_base64 = await context.image_to_base64(self.image_style_url)
        arguments = {
            "enable_pbr": self.enable_pbr,
            "text_style_prompt": self.text_style_prompt,
            "enable_safety_checker": self.enable_safety_checker,
            "enable_original_uv": self.enable_original_uv,
            "model_url": self.model_url,
            "image_style_url": f"data:image/png;base64,{image_style_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/meshy/v5/retexture",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["enable_pbr", "text_style_prompt", "enable_safety_checker", "enable_original_uv", "model_url"]

class MeshyV5Remesh(FALNode):
    """
    Meshy-5 remesh allows you to remesh and export existing 3D models into various formats
    3d, editing, transformation, modeling

    Use cases:
    - 3D model editing and refinement
    - Mesh optimization
    - Texture application
    - 3D format conversion
    - Model retopology
    """

    class Topology(Enum):
        """
        Specify the topology of the generated model. Quad for smooth surfaces, Triangle for detailed geometry.
        """
        QUAD = "quad"
        TRIANGLE = "triangle"

    class OriginAt(Enum):
        """
        Position of the origin. None means no effect.
        """
        BOTTOM = "bottom"
        CENTER = "center"


    resize_height: float = Field(
        default=0, description="Resize the model to a certain height measured in meters. Set to 0 for no resizing."
    )
    topology: Topology = Field(
        default=Topology.TRIANGLE, description="Specify the topology of the generated model. Quad for smooth surfaces, Triangle for detailed geometry."
    )
    target_polycount: int = Field(
        default=30000, description="Target number of polygons in the generated model. Actual count may vary based on geometry complexity."
    )
    model_url: str = Field(
        default="", description="URL or base64 data URI of a 3D model to remesh. Supports .glb, .gltf, .obj, .fbx, .stl formats. Can be a publicly accessible URL or data URI with MIME type application/octet-stream."
    )
    origin_at: OriginAt | None = Field(
        default=None, description="Position of the origin. None means no effect."
    )
    target_formats: list[str] = Field(
        default=[], description="List of target formats for the remeshed model."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "resize_height": self.resize_height,
            "topology": self.topology.value,
            "target_polycount": self.target_polycount,
            "model_url": self.model_url,
            "origin_at": self.origin_at.value if self.origin_at else None,
            "target_formats": self.target_formats,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/meshy/v5/remesh",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["resize_height", "topology", "target_polycount", "model_url", "origin_at"]

class HunyuanPart(FALNode):
    """
    Use the capabilities of hunyuan part to generate point clouds from your 3D files.
    3d, editing, transformation, modeling

    Use cases:
    - 3D model editing and refinement
    - Mesh optimization
    - Texture application
    - 3D format conversion
    - Model retopology
    """

    point_prompt_x: float = Field(
        default=0, description="X coordinate of the point prompt for segmentation (normalized space -1 to 1)."
    )
    point_prompt_z: float = Field(
        default=0, description="Z coordinate of the point prompt for segmentation (normalized space -1 to 1)."
    )
    use_normal: bool = Field(
        default=True, description="Whether to use normal information for segmentation."
    )
    noise_std: float = Field(
        default=0, description="Standard deviation of noise to add to sampled points."
    )
    point_num: int = Field(
        default=100000, description="Number of points to sample from the mesh."
    )
    model_file_url: str = Field(
        default="", description="URL of the 3D model file (.glb or .obj) to process for segmentation."
    )
    point_prompt_y: float = Field(
        default=0, description="Y coordinate of the point prompt for segmentation (normalized space -1 to 1)."
    )
    seed: int = Field(
        default=-1, description="The same seed and input will produce the same segmentation results."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "point_prompt_x": self.point_prompt_x,
            "point_prompt_z": self.point_prompt_z,
            "use_normal": self.use_normal,
            "noise_std": self.noise_std,
            "point_num": self.point_num,
            "model_file_url": self.model_file_url,
            "point_prompt_y": self.point_prompt_y,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan-part",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["point_prompt_x", "point_prompt_z", "use_normal", "noise_std", "point_num"]