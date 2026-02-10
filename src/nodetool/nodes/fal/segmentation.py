from pydantic import Field

from nodetool.metadata.types import ImageRef, VideoRef
from nodetool.nodes.fal.types import ControlNet, Track
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class SAM2Image(FALNode):
    """
    SAM 2 Image segments objects in images with high accuracy.
    segmentation, sam, image, masks

    Use cases:
    - Segment objects in images
    - Create object masks
    - Enable object selection
    - Generate cutouts
    - Create selection masks
    """

    image: ImageRef = Field(default=ImageRef(), description="The image to segment")
    point_coords: list[list[float]] = Field(
        default=[], description="Point coordinates for prompts [[x, y], ...]"
    )
    point_labels: list[int] = Field(
        default=[], description="Labels for points (1=foreground, 0=background)"
    )

    async def process(self, context: ProcessingContext) -> dict:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
        }
        if self.point_coords:
            arguments["point_coords"] = self.point_coords
        if self.point_labels:
            arguments["point_labels"] = self.point_labels

        res = await self.submit_request(
            context=context,
            application="fal-ai/sam2/image",
            arguments=arguments,
        )
        return {
            "masks": res.get("masks", []),
        }

    @classmethod
    def get_basic_fields(cls):
        return ["image"]

    @classmethod
    def return_type(cls):
        return {"masks": list}


class SAM2Video(FALNode):
    """
    SAM 2 Video segments and tracks objects across video frames.
    segmentation, sam, video, tracking

    Use cases:
    - Track objects in videos
    - Create video masks
    - Segment moving objects
    - Generate video cutouts
    - Enable video object selection
    """

    video: VideoRef = Field(default=VideoRef(), description="The video to segment")
    point_coords: list[list[float]] = Field(
        default=[], description="Point coordinates for prompts [[x, y], ...]"
    )
    point_labels: list[int] = Field(
        default=[], description="Labels for points (1=foreground, 0=background)"
    )

    async def process(self, context: ProcessingContext) -> dict:
        client = await self.get_client(context)
        video_bytes = await context.asset_to_bytes(self.video)
        video_url = await client.upload(video_bytes, "video/mp4")

        arguments = {
            "video_url": video_url,
        }
        if self.point_coords:
            arguments["point_coords"] = self.point_coords
        if self.point_labels:
            arguments["point_labels"] = self.point_labels

        res = await self.submit_request(
            context=context,
            application="fal-ai/sam2/video",
            arguments=arguments,
        )
        return {
            "masks_video": res.get("masks_video", {}).get("url", ""),
        }

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

    @classmethod
    def return_type(cls):
        return {"masks_video": str}


class SAM3Image(FALNode):
    """
    SAM 3 Image provides advanced segmentation with improved accuracy.
    segmentation, sam3, image, masks, advanced

    Use cases:
    - High-accuracy object segmentation
    - Complex scene segmentation
    - Precise mask generation
    - Advanced object selection
    - Detailed cutout creation
    """

    image: ImageRef = Field(default=ImageRef(), description="The image to segment")
    point_coords: list[list[float]] = Field(
        default=[], description="Point coordinates for prompts [[x, y], ...]"
    )
    point_labels: list[int] = Field(
        default=[], description="Labels for points (1=foreground, 0=background)"
    )

    async def process(self, context: ProcessingContext) -> dict:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
        }
        if self.point_coords:
            arguments["point_coords"] = self.point_coords
        if self.point_labels:
            arguments["point_labels"] = self.point_labels

        res = await self.submit_request(
            context=context,
            application="fal-ai/sam-3/image",
            arguments=arguments,
        )
        return {
            "masks": res.get("masks", []),
        }

    @classmethod
    def get_basic_fields(cls):
        return ["image"]

    @classmethod
    def return_type(cls):
        return {"masks": list}


class ImagePreprocessorDepthAnythingV2(FALNode):
    """
    Depth Anything V2 generates high-quality depth maps from images.
    depth, preprocessor, depth-map, estimation

    Use cases:
    - Generate accurate depth maps
    - Enable depth-aware effects
    - Create 3D visualizations
    - Prepare ControlNet inputs
    - Analyze image depth
    """

    image: ImageRef = Field(default=ImageRef(), description="The image to process")

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/image-preprocessors/depth-anything/v2",
            arguments=arguments,
        )
        assert res["image"] is not None
        return ImageRef(uri=res["image"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image"]


class DWPose(FALNode):
    """
    DWPose detects human poses and keypoints in images.
    pose, detection, keypoints, human

    Use cases:
    - Detect human poses
    - Extract body keypoints
    - Enable pose-guided generation
    - Analyze body positions
    - Create pose references
    """

    image: ImageRef = Field(default=ImageRef(), description="The image to analyze")

    async def process(self, context: ProcessingContext) -> dict:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/dwpose",
            arguments=arguments,
        )
        return {
            "image": ImageRef(uri=res["image"]["url"]) if "image" in res else None,
            "poses": res.get("poses", []),
        }

    @classmethod
    def get_basic_fields(cls):
        return ["image"]

    @classmethod
    def return_type(cls):
        return {"image": ImageRef, "poses": list}


class MarigoldDepth(FALNode):
    """
    Marigold Depth generates high-quality monocular depth maps.
    depth, marigold, depth-map, estimation

    Use cases:
    - Generate precise depth maps
    - Create depth visualizations
    - Enable depth-based effects
    - Prepare 3D conversions
    - Analyze scene depth
    """

    image: ImageRef = Field(default=ImageRef(), description="The image to process")

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/imageutils/marigold-depth",
            arguments=arguments,
        )
        assert res["image"] is not None
        return ImageRef(uri=res["image"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image"]
