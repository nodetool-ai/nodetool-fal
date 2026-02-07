from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.metadata.types import ImageRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


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


class Topology(Enum):
    """
    Specify the topology of the generated model. Quad for smooth surfaces, Triangle for detailed geometry.
    """
    QUAD = "quad"
    TRIANGLE = "triangle"


class SymmetryMode(Enum):
    """
    Controls symmetry behavior during model generation.
    """
    OFF = "off"
    AUTO = "auto"
    ON = "on"


class QualityMeshOption(Enum):
    """
    Combined quality and mesh type selection. Quad = smooth surfaces, Triangle = detailed geometry. These corresponds to `mesh_mode` (if the option contains 'Triangle', mesh_mode is 'Raw', otherwise 'Quad') and `quality_override` (the numeric part of the option) parameters in Hyper3D API.
    """
    VALUE_4K_QUAD = "4K Quad"
    VALUE_8K_QUAD = "8K Quad"
    VALUE_18K_QUAD = "18K Quad"
    VALUE_50K_QUAD = "50K Quad"
    VALUE_2K_TRIANGLE = "2K Triangle"
    VALUE_20K_TRIANGLE = "20K Triangle"
    VALUE_150K_TRIANGLE = "150K Triangle"
    VALUE_500K_TRIANGLE = "500K Triangle"


class GeometryFileFormat(Enum):
    """
    Format of the geometry file. Possible values: glb, usdz, fbx, obj, stl. Default is glb.
    """
    GLB = "glb"
    USDZ = "usdz"
    FBX = "fbx"
    OBJ = "obj"
    STL = "stl"


class Addons(Enum):
    """
    The HighPack option will provide 4K resolution textures instead of the default 1K, as well as models with high-poly. It will cost **triple the billable units**.
    """
    HIGHPACK = "HighPack"


class Material(Enum):
    """
    Material type. PBR: Physically-based materials with realistic lighting. Shaded: Simple materials with baked lighting. All: Both types included.
    """
    PBR = "PBR"
    SHADED = "Shaded"
    ALL = "All"


class Hunyuan3DV3SketchTo3D(FALNode):
    """
    Hunyuan3d V3
    3d, generation, image-to-3d, modeling

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    input_image_url: ImageRef = Field(
        default=ImageRef(), description="URL of sketch or line art image to transform into a 3D model. Image resolution must be between 128x128 and 5000x5000 pixels."
    )
    prompt: str = Field(
        default="", description="Text prompt describing the 3D content attributes such as color, category, and material."
    )
    face_count: int = Field(
        default=500000, description="Target face count. Range: 40000-1500000"
    )
    enable_pbr: bool = Field(
        default=False, description="Whether to enable PBR material generation."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        input_image_url_base64 = await context.image_to_base64(self.input_image_url)
        arguments = {
            "input_image_url": f"data:image/png;base64,{input_image_url_base64}",
            "prompt": self.prompt,
            "face_count": self.face_count,
            "enable_pbr": self.enable_pbr,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan3d-v3/sketch-to-3d",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["input_image_url", "prompt", "face_count", "enable_pbr"]



class Hunyuan3DV3ImageTo3D(FALNode):
    """
    Hunyuan3d V3
    3d, generation, image-to-3d, modeling

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    input_image_url: ImageRef = Field(
        default=ImageRef(), description="URL of image to use while generating the 3D model."
    )
    polygon_type: PolygonType = Field(
        default=PolygonType.TRIANGLE, description="Polygon type. Only takes effect when GenerateType is LowPoly."
    )
    face_count: int = Field(
        default=500000, description="Target face count. Range: 40000-1500000"
    )
    right_image_url: ImageRef = Field(
        default=ImageRef(), description="Optional right view image URL for better 3D reconstruction."
    )
    back_image_url: ImageRef = Field(
        default=ImageRef(), description="Optional back view image URL for better 3D reconstruction."
    )
    enable_pbr: bool = Field(
        default=False, description="Whether to enable PBR material generation. Does not take effect when generate_type is Geometry."
    )
    generate_type: GenerateType = Field(
        default=GenerateType.NORMAL, description="Generation type. Normal: textured model. LowPoly: polygon reduction. Geometry: white model without texture."
    )
    left_image_url: ImageRef = Field(
        default=ImageRef(), description="Optional left view image URL for better 3D reconstruction."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        input_image_url_base64 = await context.image_to_base64(self.input_image_url)
        right_image_url_base64 = await context.image_to_base64(self.right_image_url)
        back_image_url_base64 = await context.image_to_base64(self.back_image_url)
        left_image_url_base64 = await context.image_to_base64(self.left_image_url)
        arguments = {
            "input_image_url": f"data:image/png;base64,{input_image_url_base64}",
            "polygon_type": self.polygon_type.value,
            "face_count": self.face_count,
            "right_image_url": f"data:image/png;base64,{right_image_url_base64}",
            "back_image_url": f"data:image/png;base64,{back_image_url_base64}",
            "enable_pbr": self.enable_pbr,
            "generate_type": self.generate_type.value,
            "left_image_url": f"data:image/png;base64,{left_image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan3d-v3/image-to-3d",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["input_image_url", "polygon_type", "face_count", "right_image_url", "back_image_url"]

class Sam33DBody(FALNode):
    """
    Sam 3
    3d, generation, image-to-3d, modeling

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image containing humans"
    )
    include_3d_keypoints: bool = Field(
        default=True, description="Include 3D keypoint markers (spheres) in the GLB mesh for visualization"
    )
    mask_url: ImageRef = Field(
        default=ImageRef(), description="Optional URL of a binary mask image (white=person, black=background). When provided, skips auto human detection and uses this mask instead. Bbox is auto-computed from the mask."
    )
    export_meshes: bool = Field(
        default=True, description="Export individual mesh files (.ply) per person"
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_url_base64 = await context.image_to_base64(self.image_url)
        mask_url_base64 = await context.image_to_base64(self.mask_url)
        arguments = {
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "include_3d_keypoints": self.include_3d_keypoints,
            "mask_url": f"data:image/png;base64,{mask_url_base64}",
            "export_meshes": self.export_meshes,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sam-3/3d-body",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["image_url", "include_3d_keypoints", "mask_url", "export_meshes"]

class Sam33DObjects(FALNode):
    """
    Sam 3
    3d, generation, image-to-3d, modeling

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    prompt: str = Field(
        default="car", description="Text prompt for auto-segmentation when no masks provided (e.g., 'chair', 'lamp')"
    )
    export_textured_glb: bool = Field(
        default=False, description="If True, exports GLB with baked texture and UVs instead of vertex colors."
    )
    detection_threshold: float = Field(
        default=0.0, description="Detection confidence threshold (0.1-1.0). Lower = more detections but less precise. If not set, uses the model's default."
    )
    pointmap_url: str = Field(
        default="", description="Optional URL to external pointmap/depth data (NPY or NPZ format) for improved 3D reconstruction depth estimation"
    )
    box_prompts: list[str] = Field(
        default=[], description="Box prompts for auto-segmentation when no masks provided. Multiple boxes supported - each produces a separate object mask for 3D reconstruction."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image to reconstruct in 3D"
    )
    mask_urls: list[str] = Field(
        default=[], description="Optional list of mask URLs (one per object). If not provided, use prompt/point_prompts/box_prompts to auto-segment, or entire image will be used."
    )
    point_prompts: list[str] = Field(
        default=[], description="Point prompts for auto-segmentation when no masks provided"
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility"
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "export_textured_glb": self.export_textured_glb,
            "detection_threshold": self.detection_threshold,
            "pointmap_url": self.pointmap_url,
            "box_prompts": self.box_prompts,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "mask_urls": self.mask_urls,
            "point_prompts": self.point_prompts,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sam-3/3d-objects",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "export_textured_glb", "detection_threshold", "pointmap_url", "box_prompts"]

class Omnipart(FALNode):
    """
    Omnipart
    3d, generation, image-to-3d, modeling

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    input_image_url: ImageRef = Field(
        default=ImageRef(), description="URL of image to use while generating the 3D model."
    )
    parts: str = Field(
        default="", description="Specify which segments to merge (e.g., '0,1;3,4' merges segments 0&1 together and 3&4 together)"
    )
    seed: int = Field(
        default=765464, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    minimum_segment_size: int = Field(
        default=2000, description="Minimum segment size (pixels) for the model."
    )
    guidance_scale: float = Field(
        default=7.5, description="Guidance scale for the model."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        input_image_url_base64 = await context.image_to_base64(self.input_image_url)
        arguments = {
            "input_image_url": f"data:image/png;base64,{input_image_url_base64}",
            "parts": self.parts,
            "seed": self.seed,
            "minimum_segment_size": self.minimum_segment_size,
            "guidance_scale": self.guidance_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/omnipart",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["input_image_url", "parts", "seed", "minimum_segment_size", "guidance_scale"]

class BytedanceSeed3DImageTo3D(FALNode):
    """
    Bytedance
    3d, generation, image-to-3d, modeling

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image for the 3D asset generation."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/bytedance/seed3d/image-to-3d",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["image_url"]



class MeshyV5MultiImageTo3D(FALNode):
    """
    Meshy 5 Multi
    3d, generation, image-to-3d, modeling

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    enable_pbr: bool = Field(
        default=False, description="Generate PBR Maps (metallic, roughness, normal) in addition to base color. Requires should_texture to be true."
    )
    should_texture: bool = Field(
        default=True, description="Whether to generate textures. False provides mesh without textures for 5 credits, True adds texture generation for additional 10 credits."
    )
    target_polycount: int = Field(
        default=30000, description="Target number of polygons in the generated model"
    )
    is_a_t_pose: bool = Field(
        default=False, description="Whether to generate the model in an A/T pose"
    )
    texture_image_url: ImageRef = Field(
        default=ImageRef(), description="2D image to guide the texturing process. Requires should_texture to be true."
    )
    topology: Topology = Field(
        default=Topology.TRIANGLE, description="Specify the topology of the generated model. Quad for smooth surfaces, Triangle for detailed geometry."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, input data will be checked for safety before processing."
    )
    symmetry_mode: SymmetryMode = Field(
        default=SymmetryMode.AUTO, description="Controls symmetry behavior during model generation."
    )
    image_urls: list[str] = Field(
        default=[], description="1 to 4 images for 3D model creation. All images should depict the same object from different angles. Supports .jpg, .jpeg, .png formats, and AVIF/HEIF which will be automatically converted. If more than 4 images are provided, only the first 4 will be used."
    )
    texture_prompt: str = Field(
        default="", description="Text prompt to guide the texturing process. Requires should_texture to be true."
    )
    should_remesh: bool = Field(
        default=True, description="Whether to enable the remesh phase. When false, returns triangular mesh ignoring topology and target_polycount."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        texture_image_url_base64 = await context.image_to_base64(self.texture_image_url)
        arguments = {
            "enable_pbr": self.enable_pbr,
            "should_texture": self.should_texture,
            "target_polycount": self.target_polycount,
            "is_a_t_pose": self.is_a_t_pose,
            "texture_image_url": f"data:image/png;base64,{texture_image_url_base64}",
            "topology": self.topology.value,
            "enable_safety_checker": self.enable_safety_checker,
            "symmetry_mode": self.symmetry_mode.value,
            "image_urls": self.image_urls,
            "texture_prompt": self.texture_prompt,
            "should_remesh": self.should_remesh,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/meshy/v5/multi-image-to-3d",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["enable_pbr", "should_texture", "target_polycount", "is_a_t_pose", "texture_image_url"]



class MeshyV6PreviewImageTo3D(FALNode):
    """
    Meshy 6 Preview
    3d, generation, image-to-3d, modeling

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    enable_pbr: bool = Field(
        default=False, description="Generate PBR Maps (metallic, roughness, normal) in addition to base color"
    )
    is_a_t_pose: bool = Field(
        default=False, description="Whether to generate the model in an A/T pose"
    )
    target_polycount: int = Field(
        default=30000, description="Target number of polygons in the generated model"
    )
    should_texture: bool = Field(
        default=True, description="Whether to generate textures"
    )
    texture_image_url: ImageRef = Field(
        default=ImageRef(), description="2D image to guide the texturing process"
    )
    topology: Topology = Field(
        default=Topology.TRIANGLE, description="Specify the topology of the generated model. Quad for smooth surfaces, Triangle for detailed geometry."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="Image URL or base64 data URI for 3D model creation. Supports .jpg, .jpeg, and .png formats. Also supports AVIF and HEIF formats which will be automatically converted."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, input data will be checked for safety before processing."
    )
    symmetry_mode: SymmetryMode = Field(
        default=SymmetryMode.AUTO, description="Controls symmetry behavior during model generation. Off disables symmetry, Auto determines it automatically, On enforces symmetry."
    )
    texture_prompt: str = Field(
        default="", description="Text prompt to guide the texturing process"
    )
    should_remesh: bool = Field(
        default=True, description="Whether to enable the remesh phase"
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        texture_image_url_base64 = await context.image_to_base64(self.texture_image_url)
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "enable_pbr": self.enable_pbr,
            "is_a_t_pose": self.is_a_t_pose,
            "target_polycount": self.target_polycount,
            "should_texture": self.should_texture,
            "texture_image_url": f"data:image/png;base64,{texture_image_url_base64}",
            "topology": self.topology.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "enable_safety_checker": self.enable_safety_checker,
            "symmetry_mode": self.symmetry_mode.value,
            "texture_prompt": self.texture_prompt,
            "should_remesh": self.should_remesh,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/meshy/v6-preview/image-to-3d",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["enable_pbr", "is_a_t_pose", "target_polycount", "should_texture", "texture_image_url"]





class Hyper3DRodinV2(FALNode):
    """
    Hyper3d
    3d, generation, image-to-3d, modeling

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    quality_mesh_option: QualityMeshOption = Field(
        default=QualityMeshOption.VALUE_500K_TRIANGLE, description="Combined quality and mesh type selection. Quad = smooth surfaces, Triangle = detailed geometry. These corresponds to `mesh_mode` (if the option contains 'Triangle', mesh_mode is 'Raw', otherwise 'Quad') and `quality_override` (the numeric part of the option) parameters in Hyper3D API."
    )
    prompt: str = Field(
        default="", description="A textual prompt to guide model generation. Optional for Image-to-3D mode - if empty, AI will generate a prompt based on your images."
    )
    preview_render: bool = Field(
        default=False, description="Generate a preview render image of the 3D model along with the model files."
    )
    bbox_condition: list[int] = Field(
        default=[], description="An array that specifies the bounding box dimensions [width, height, length]."
    )
    TAPose: bool = Field(
        default=False, description="Generate characters in T-pose or A-pose format, making them easier to rig and animate in 3D software."
    )
    input_image_urls: list[str] = Field(
        default=[], description="URL of images to use while generating the 3D model. Required for Image-to-3D mode. Up to 5 images allowed."
    )
    use_original_alpha: bool = Field(
        default=False, description="When enabled, preserves the transparency channel from input images during 3D generation."
    )
    geometry_file_format: GeometryFileFormat = Field(
        default=GeometryFileFormat.GLB, description="Format of the geometry file. Possible values: glb, usdz, fbx, obj, stl. Default is glb."
    )
    addons: Addons | None = Field(
        default=None, description="The HighPack option will provide 4K resolution textures instead of the default 1K, as well as models with high-poly. It will cost **triple the billable units**."
    )
    seed: int = Field(
        default=-1, description="Seed value for randomization, ranging from 0 to 65535. Optional."
    )
    material: Material = Field(
        default=Material.ALL, description="Material type. PBR: Physically-based materials with realistic lighting. Shaded: Simple materials with baked lighting. All: Both types included."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "quality_mesh_option": self.quality_mesh_option.value,
            "prompt": self.prompt,
            "preview_render": self.preview_render,
            "bbox_condition": self.bbox_condition,
            "TAPose": self.TAPose,
            "input_image_urls": self.input_image_urls,
            "use_original_alpha": self.use_original_alpha,
            "geometry_file_format": self.geometry_file_format.value,
            "addons": self.addons.value if self.addons else None,
            "seed": self.seed,
            "material": self.material.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/hyper3d/rodin/v2",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["quality_mesh_option", "prompt", "preview_render", "bbox_condition", "TAPose"]

class Pshuman(FALNode):
    """
    Pshuman
    3d, generation, image-to-3d, modeling

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    guidance_scale: float = Field(
        default=4, description="Guidance scale for the diffusion process. Controls how much the output adheres to the generated views."
    )
    seed: int = Field(
        default=-1, description="Seed for reproducibility. If None, a random seed will be used."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="A direct URL to the input image of a person."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pshuman",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["guidance_scale", "seed", "image_url"]

class Hunyuan_WorldImageToWorld(FALNode):
    """
    Hunyuan World
    3d, generation, image-to-3d, modeling

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    classes: str = Field(
        default="", description="Classes to use for the world generation."
    )
    export_drc: bool = Field(
        default=False, description="Whether to export DRC (Dynamic Resource Configuration)."
    )
    labels_fg1: str = Field(
        default="", description="Labels for the first foreground object."
    )
    labels_fg2: str = Field(
        default="", description="Labels for the second foreground object."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to convert to a world."
    )

    async def process(self, context: ProcessingContext) -> Any:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "classes": self.classes,
            "export_drc": self.export_drc,
            "labels_fg1": self.labels_fg1,
            "labels_fg2": self.labels_fg2,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan_world/image-to-world",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["classes", "export_drc", "labels_fg1", "labels_fg2", "image_url"]