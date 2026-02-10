from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.metadata.types import VideoRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class OpenrouterRouterVideoEnterprise(FALNode):
    """
    Run any VLM (Video Language Model) with fal, powered by OpenRouter.
    video, transcription, analysis, video-understanding

    Use cases:
    - Video transcription
    - Video content analysis
    - Automated captioning
    - Video understanding
    - Content indexing
    """

    prompt: str = Field(
        default="", description="Prompt to be used for the video processing"
    )
    video_urls: VideoRef = Field(
        default=VideoRef(), description="List of URLs or data URIs of video files to process. Supported formats: mp4, mpeg, mov, webm. For Google Gemini on AI Studio, YouTube links are also supported. Mutually exclusive with video_url."
    )
    reasoning: bool = Field(
        default=False, description="Should reasoning be the part of the final answer."
    )
    system_prompt: str = Field(
        default="", description="System prompt to provide context or instructions to the model"
    )
    model: str = Field(
        default="", description="Name of the model to use. Charged based on actual token usage."
    )
    max_tokens: str = Field(
        default="", description="This sets the upper limit for the number of tokens the model can generate in response. It won't produce more than this limit. The maximum value is the context length minus the prompt length."
    )
    temperature: float = Field(
        default=1, description="This setting influences the variety in the model's responses. Lower values lead to more predictable and typical responses, while higher values encourage more diverse and less common responses. At 0, the model always gives the same response for a given input."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "prompt": self.prompt,
            "video_urls": self.video_urls,
            "reasoning": self.reasoning,
            "system_prompt": self.system_prompt,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="openrouter/router/video/enterprise",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "video_urls", "reasoning", "system_prompt", "model"]

class OpenrouterRouterVideo(FALNode):
    """
    Run any VLM (Video Language Model) with fal, powered by OpenRouter.
    video, transcription, analysis, video-understanding

    Use cases:
    - Video transcription
    - Video content analysis
    - Automated captioning
    - Video understanding
    - Content indexing
    """

    prompt: str = Field(
        default="", description="Prompt to be used for the video processing"
    )
    video_urls: VideoRef = Field(
        default=VideoRef(), description="List of URLs or data URIs of video files to process. Supported formats: mp4, mpeg, mov, webm. For Google Gemini on AI Studio, YouTube links are also supported. Mutually exclusive with video_url."
    )
    reasoning: bool = Field(
        default=False, description="Should reasoning be the part of the final answer."
    )
    system_prompt: str = Field(
        default="", description="System prompt to provide context or instructions to the model"
    )
    model: str = Field(
        default="", description="Name of the model to use. Charged based on actual token usage."
    )
    max_tokens: str = Field(
        default="", description="This sets the upper limit for the number of tokens the model can generate in response. It won't produce more than this limit. The maximum value is the context length minus the prompt length."
    )
    temperature: float = Field(
        default=1, description="This setting influences the variety in the model's responses. Lower values lead to more predictable and typical responses, while higher values encourage more diverse and less common responses. At 0, the model always gives the same response for a given input."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "prompt": self.prompt,
            "video_urls": self.video_urls,
            "reasoning": self.reasoning,
            "system_prompt": self.system_prompt,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="openrouter/router/video",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "video_urls", "reasoning", "system_prompt", "model"]