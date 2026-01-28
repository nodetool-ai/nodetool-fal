from pydantic import Field

from nodetool.metadata.types import ImageRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


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
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> dict:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/trellis",
            arguments=arguments,
        )
        return {
            "model_url": res.get("model", {}).get("url", ""),
            "glb_url": res.get("glb", {}).get("url", ""),
        }

    @classmethod
    def get_basic_fields(cls):
        return ["image"]

    @classmethod
    def return_type(cls):
        return {"model_url": str, "glb_url": str}


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
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> dict:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan3d/v2",
            arguments=arguments,
        )
        return {
            "model_url": res.get("model", {}).get("url", ""),
            "glb_url": res.get("glb", {}).get("url", ""),
        }

    @classmethod
    def get_basic_fields(cls):
        return ["image"]

    @classmethod
    def return_type(cls):
        return {"model_url": str, "glb_url": str}


class TripoSR(FALNode):
    """
    TripoSR generates 3D models from single images with fast inference.
    3d, generation, image-to-3d, triposr, fast

    Use cases:
    - Quick 3D model generation
    - Fast prototyping
    - Create 3D assets rapidly
    - Real-time 3D conversion
    - Batch 3D generation
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to convert to 3D"
    )

    async def process(self, context: ProcessingContext) -> dict:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/triposr",
            arguments=arguments,
        )
        return {
            "model_url": res.get("model", {}).get("url", ""),
        }

    @classmethod
    def get_basic_fields(cls):
        return ["image"]

    @classmethod
    def return_type(cls):
        return {"model_url": str}


class Era3D(FALNode):
    """
    Era-3D generates multi-view images and 3D models from single images.
    3d, generation, image-to-3d, era3d, multiview

    Use cases:
    - Generate multi-view images
    - Create 3D from single photo
    - Produce 3D content
    - Generate 3D assets
    - Create 3D visualizations
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to convert to 3D"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> dict:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/era-3d",
            arguments=arguments,
        )
        return {
            "images": [ImageRef(uri=img["url"]) for img in res.get("images", [])],
            "model_url": res.get("model", {}).get("url", ""),
        }

    @classmethod
    def get_basic_fields(cls):
        return ["image"]

    @classmethod
    def return_type(cls):
        return {"images": list, "model_url": str}
