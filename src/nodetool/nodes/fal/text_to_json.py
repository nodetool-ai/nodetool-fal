from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.metadata.types import ImageRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class BriaFiboEditEditStructured_instruction(FALNode):
    """
    Structured Instructions Generation endpoint for Fibo Edit, Bria's newest editing model.
    text, analysis, json, extraction

    Use cases:
    - Text analysis to structured data
    - Content extraction
    - Data structuring
    - Information extraction
    - Text classification
    """

    sync_mode: bool = Field(
        default=False, description="If true, returns the image directly in the response (increases latency)."
    )
    seed: int = Field(
        default=5555, description="Random seed for reproducibility."
    )
    mask: ImageRef = Field(
        default=ImageRef(), description="Reference image mask (file or URL). Optional."
    )
    instruction: str = Field(
        default="", description="Instruction for image editing."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Reference image (file or URL)."
    )

    async def process(self, context: ProcessingContext) -> Any:
        mask_base64 = await context.image_to_base64(self.mask)
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "sync_mode": self.sync_mode,
            "seed": self.seed,
            "mask_url": f"data:image/png;base64,{mask_base64}",
            "instruction": self.instruction,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/fibo-edit/edit/structured_instruction",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["sync_mode", "seed", "mask", "instruction", "image"]

class BriaFiboLiteGenerateStructured_prompt(FALNode):
    """
    Structured Prompt Generation endpoint for Fibo-Lite, Bria's SOTA Open source model
    text, analysis, json, extraction

    Use cases:
    - Text analysis to structured data
    - Content extraction
    - Data structuring
    - Information extraction
    - Text classification
    """

    prompt: str = Field(
        default="", description="Prompt for image generation."
    )
    seed: int = Field(
        default=5555, description="Random seed for reproducibility."
    )
    structured_prompt: str = Field(
        default="", description="The structured prompt to generate an image from."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Reference image (file or URL)."
    )

    async def process(self, context: ProcessingContext) -> Any:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "seed": self.seed,
            "structured_prompt": self.structured_prompt,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/fibo-lite/generate/structured_prompt",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "seed", "structured_prompt", "image"]

class BriaFiboLiteGenerateStructured_promptLite(FALNode):
    """
    Structured Prompt Generation endpoint for Fibo-Lite, Bria's SOTA Open source model
    text, analysis, json, extraction

    Use cases:
    - Text analysis to structured data
    - Content extraction
    - Data structuring
    - Information extraction
    - Text classification
    """

    prompt: str = Field(
        default="", description="Prompt for image generation."
    )
    seed: int = Field(
        default=5555, description="Random seed for reproducibility."
    )
    structured_prompt: str = Field(
        default="", description="The structured prompt to generate an image from."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Reference image (file or URL)."
    )

    async def process(self, context: ProcessingContext) -> Any:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "seed": self.seed,
            "structured_prompt": self.structured_prompt,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/fibo-lite/generate/structured_prompt/lite",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "seed", "structured_prompt", "image"]

class BriaFiboGenerateStructured_prompt(FALNode):
    """
    Structured Prompt Generation endpoint for Fibo, Bria's SOTA Open source model
    text, analysis, json, extraction

    Use cases:
    - Text analysis to structured data
    - Content extraction
    - Data structuring
    - Information extraction
    - Text classification
    """

    prompt: str = Field(
        default="", description="Prompt for image generation."
    )
    seed: int = Field(
        default=5555, description="Random seed for reproducibility."
    )
    structured_prompt: str = Field(
        default="", description="The structured prompt to generate an image from."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Reference image (file or URL)."
    )

    async def process(self, context: ProcessingContext) -> Any:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "seed": self.seed,
            "structured_prompt": self.structured_prompt,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/fibo/generate/structured_prompt",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "seed", "structured_prompt", "image"]