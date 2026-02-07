from pydantic import Field
from typing import Any
from nodetool.metadata.types import ImageRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class BagelUnderstand(FALNode):
    """
    Bagel is a 7B parameter multimodal model from Bytedance-Seed that can generate both text and images.
    vision, analysis, json, image-understanding

    Use cases:
    - Image analysis to structured data
    - Visual content understanding
    - Automated image metadata extraction
    - Content classification
    - Image-based data extraction
    """

    prompt: str = Field(
        default="", description="The prompt to query the image with."
    )
    seed: int = Field(
        default=-1, description="The seed to use for the generation."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The image for the query."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "seed": self.seed,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/bagel/understand",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "seed", "image_url"]