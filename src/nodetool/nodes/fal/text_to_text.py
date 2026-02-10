from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class HalfMoonAiAiDetectorDetectText(FALNode):
    """
    AI Detector (Text) is an advanced AI service that analyzes a passage and returns a verdict on whether it was likely written by AI.
    text, processing, transformation, nlp

    Use cases:
    - Text transformation
    - Content analysis
    - Text classification
    - Language processing
    - Content detection
    """

    text: str = Field(
        default="", description="Text content to analyze for AI generation."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "text": self.text,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="half-moon-ai/ai-detector/detect-text",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["text"]