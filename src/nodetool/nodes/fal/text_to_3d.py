from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.metadata.types import ImageRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class HunyuanMotionFast(FALNode):
    """
    Generate 3D human motions via text-to-generation interface of Hunyuan Motion!
    3d, generation, text-to-3d, modeling, fast

    Use cases:
    - 3D model generation from text
    - Concept visualization
    - Game asset creation
    - Architectural prototyping
    - Product design visualization
    """

    class OutputFormat(Enum):
        """
        Output format: 'fbx' for animation files, 'dict' for raw JSON.
        """
        FBX = "fbx"
        DICT = "dict"


    prompt: str = Field(
        default="", description="Text prompt describing the motion to generate."
    )
    duration: float = Field(
        default=5, description="Motion duration in seconds (0.5-12.0)."
    )
    guidance_scale: float = Field(
        default=5, description="Classifier-free guidance scale. Higher = more faithful to prompt."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducible generation."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.FBX, description="Output format: 'fbx' for animation files, 'dict' for raw JSON."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "output_format": self.output_format.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan-motion/fast",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "duration", "guidance_scale", "seed", "output_format"]

class HunyuanMotion(FALNode):
    """
    Generate 3D human motions via text-to-generation interface of Hunyuan Motion!
    3d, generation, text-to-3d, modeling

    Use cases:
    - 3D model generation from text
    - Concept visualization
    - Game asset creation
    - Architectural prototyping
    - Product design visualization
    """

    class OutputFormat(Enum):
        """
        Output format: 'fbx' for animation files, 'dict' for raw JSON.
        """
        FBX = "fbx"
        DICT = "dict"


    prompt: str = Field(
        default="", description="Text prompt describing the motion to generate."
    )
    duration: float = Field(
        default=5, description="Motion duration in seconds (0.5-12.0)."
    )
    guidance_scale: float = Field(
        default=5, description="Classifier-free guidance scale. Higher = more faithful to prompt."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducible generation."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.FBX, description="Output format: 'fbx' for animation files, 'dict' for raw JSON."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "output_format": self.output_format.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan-motion",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "duration", "guidance_scale", "seed", "output_format"]

class Hunyuan3dV3TextTo3d(FALNode):
    """
    Turn simple sketches into detailed, fully-textured 3D models. Instantly convert your concept designs into formats ready for Unity, Unreal, and Blender.
    3d, generation, text-to-3d, modeling

    Use cases:
    - 3D model generation from text
    - Concept visualization
    - Game asset creation
    - Architectural prototyping
    - Product design visualization
    """

    class PolygonType(Enum):
        """
        Polygon type. Only takes effect when GenerateType is LowPoly.
        """
        TRIANGLE = "triangle"
        QUADRILATERAL = "quadrilateral"

    class GenerateType(Enum):
        """
        Generation type. Normal: textured model. LowPoly: polygon reduction. Geometry: white model without texture.
        """
        NORMAL = "Normal"
        LOWPOLY = "LowPoly"
        GEOMETRY = "Geometry"


    enable_pbr: bool = Field(
        default=False, description="Whether to enable PBR material generation"
    )
    polygon_type: PolygonType = Field(
        default=PolygonType.TRIANGLE, description="Polygon type. Only takes effect when GenerateType is LowPoly."
    )
    face_count: int = Field(
        default=500000, description="Target face count. Range: 40000-1500000"
    )
    prompt: str = Field(
        default="", description="Text description of the 3D content to generate. Supports up to 1024 UTF-8 characters."
    )
    generate_type: GenerateType = Field(
        default=GenerateType.NORMAL, description="Generation type. Normal: textured model. LowPoly: polygon reduction. Geometry: white model without texture."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "enable_pbr": self.enable_pbr,
            "polygon_type": self.polygon_type.value,
            "face_count": self.face_count,
            "prompt": self.prompt,
            "generate_type": self.generate_type.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan3d-v3/text-to-3d",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["enable_pbr", "polygon_type", "face_count", "prompt", "generate_type"]

class MeshyV6PreviewTextTo3d(FALNode):
    """
    Meshy-6-Preview is the latest model from Meshy. It generates realistic and production ready 3D models.
    3d, generation, text-to-3d, modeling

    Use cases:
    - 3D model generation from text
    - Concept visualization
    - Game asset creation
    - Architectural prototyping
    - Product design visualization
    """

    class ArtStyle(Enum):
        """
        Desired art style of the object. Note: enable_pbr should be false for sculpture style.
        """
        REALISTIC = "realistic"
        SCULPTURE = "sculpture"

    class Mode(Enum):
        """
        Generation mode. 'preview' returns untextured geometry only, 'full' returns textured model (preview + refine).
        """
        PREVIEW = "preview"
        FULL = "full"

    class SymmetryMode(Enum):
        """
        Controls symmetry behavior during model generation.
        """
        OFF = "off"
        AUTO = "auto"
        ON = "on"

    class Topology(Enum):
        """
        Specify the topology of the generated model. Quad for smooth surfaces, Triangle for detailed geometry.
        """
        QUAD = "quad"
        TRIANGLE = "triangle"


    prompt: str = Field(
        default="", description="Describe what kind of object the 3D model is. Maximum 600 characters."
    )
    enable_pbr: bool = Field(
        default=False, description="Generate PBR Maps (metallic, roughness, normal) in addition to base color. Should be false for sculpture style."
    )
    target_polycount: int = Field(
        default=30000, description="Target number of polygons in the generated model"
    )
    art_style: ArtStyle = Field(
        default=ArtStyle.REALISTIC, description="Desired art style of the object. Note: enable_pbr should be false for sculpture style."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, input data will be checked for safety before processing."
    )
    mode: Mode = Field(
        default=Mode.FULL, description="Generation mode. 'preview' returns untextured geometry only, 'full' returns textured model (preview + refine)."
    )
    symmetry_mode: SymmetryMode = Field(
        default=SymmetryMode.AUTO, description="Controls symmetry behavior during model generation."
    )
    should_remesh: bool = Field(
        default=True, description="Whether to enable the remesh phase. When false, returns unprocessed triangular mesh."
    )
    texture_image_url: ImageRef = Field(
        default=ImageRef(), description="2D image to guide the texturing process (only used in 'full' mode)"
    )
    topology: Topology = Field(
        default=Topology.TRIANGLE, description="Specify the topology of the generated model. Quad for smooth surfaces, Triangle for detailed geometry."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion. This will use a large language model to expand the prompt with additional details while maintaining the original meaning."
    )
    seed: int = Field(
        default=-1, description="Seed for reproducible results. Same prompt and seed usually generate the same result."
    )
    is_a_t_pose: bool = Field(
        default=False, description="Whether to generate the model in an A/T pose"
    )
    texture_prompt: str = Field(
        default="", description="Additional text prompt to guide the texturing process (only used in 'full' mode)"
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        texture_image_url_base64 = await context.image_to_base64(self.texture_image_url)
        arguments = {
            "prompt": self.prompt,
            "enable_pbr": self.enable_pbr,
            "target_polycount": self.target_polycount,
            "art_style": self.art_style.value,
            "enable_safety_checker": self.enable_safety_checker,
            "mode": self.mode.value,
            "symmetry_mode": self.symmetry_mode.value,
            "should_remesh": self.should_remesh,
            "texture_image_url": f"data:image/png;base64,{texture_image_url_base64}",
            "topology": self.topology.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "is_a_t_pose": self.is_a_t_pose,
            "texture_prompt": self.texture_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/meshy/v6-preview/text-to-3d",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "enable_pbr", "target_polycount", "art_style", "enable_safety_checker"]