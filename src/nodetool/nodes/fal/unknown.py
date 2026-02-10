from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.metadata.types import VideoRef, AudioRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class WorkflowUtilitiesInterleaveVideo(FALNode):
    """
    ffmpeg utility to interleave videos
    utility, processing, general

    Use cases:
    - General media processing
    - Utility operations
    - Content manipulation
    - Automated workflows
    - Data processing
    """

    video_urls: list[str] = Field(
        default=[], description="List of video URLs to interleave in order"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "video_urls": self.video_urls,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/workflow-utilities/interleave-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video_urls"]

class Qwen3TtsCloneVoice17b(FALNode):
    """
    Clone your voices using Qwen3-TTS Clone-Voice model with zero shot cloning capabilities and use it on text-to-speech models to create speeches of yours!
    utility, processing, general

    Use cases:
    - General media processing
    - Utility operations
    - Content manipulation
    - Automated workflows
    - Data processing
    """

    audio_url: AudioRef = Field(
        default=AudioRef(), description="URL to the reference audio file used for voice cloning."
    )
    reference_text: str = Field(
        default="", description="Optional reference text that was used when creating the speaker embedding. Providing this can improve synthesis quality when using a cloned voice."
    )

    async def process(self, context: ProcessingContext) -> Any:
        arguments = {
            "audio_url": self.audio_url,
            "reference_text": self.reference_text,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-3-tts/clone-voice/1.7b",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["audio_url", "reference_text"]

class Qwen3TtsCloneVoice06b(FALNode):
    """
    Clone your voices using Qwen3-TTS Clone-Voice model with zero shot cloning capabilities and use it on text-to-speech models to create speeches of yours!
    utility, processing, general

    Use cases:
    - General media processing
    - Utility operations
    - Content manipulation
    - Automated workflows
    - Data processing
    """

    audio_url: AudioRef = Field(
        default=AudioRef(), description="URL to the reference audio file used for voice cloning."
    )
    reference_text: str = Field(
        default="", description="Optional reference text that was used when creating the speaker embedding. Providing this can improve synthesis quality when using a cloned voice."
    )

    async def process(self, context: ProcessingContext) -> Any:
        arguments = {
            "audio_url": self.audio_url,
            "reference_text": self.reference_text,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-3-tts/clone-voice/0.6b",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["audio_url", "reference_text"]

class OpenrouterRouterAudio(FALNode):
    """
    Run any ALM (Audio Language Model) with fal, powered by OpenRouter.
    utility, processing, general

    Use cases:
    - General media processing
    - Utility operations
    - Content manipulation
    - Automated workflows
    - Data processing
    """

    prompt: str = Field(
        default="", description="Prompt to be used for the audio processing"
    )
    system_prompt: str = Field(
        default="", description="System prompt to provide context or instructions to the model"
    )
    reasoning: bool = Field(
        default=False, description="Should reasoning be the part of the final answer."
    )
    model: str = Field(
        default="", description="Name of the model to use. Charged based on actual token usage."
    )
    audio_url: AudioRef = Field(
        default=AudioRef(), description="URL or data URI of the audio file to process. Supported formats: wav, mp3, aiff, aac, ogg, flac, m4a."
    )
    temperature: float = Field(
        default=1, description="This setting influences the variety in the model's responses. Lower values lead to more predictable and typical responses, while higher values encourage more diverse and less common responses. At 0, the model always gives the same response for a given input."
    )
    max_tokens: int = Field(
        default=0, description="This sets the upper limit for the number of tokens the model can generate in response. It won't produce more than this limit. The maximum value is the context length minus the prompt length."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "reasoning": self.reasoning,
            "model": self.model,
            "audio_url": self.audio_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="openrouter/router/audio",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "system_prompt", "reasoning", "model", "audio_url"]