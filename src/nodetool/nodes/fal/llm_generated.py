from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class OpenRouter(FALNode):
    """
    OpenRouter provides unified access to any LLM (Large Language Model) through a single API.
    llm, chat, openrouter, multimodel, language-model

    Use cases:
    - Run any LLM through unified interface
    - Switch between models seamlessly
    - Access multiple LLM providers
    - Flexible model selection
    - Unified LLM API access
    """

    model: str = Field(
        default="", description="Name of the model to use. Charged based on actual token usage."
    )
    prompt: str = Field(
        default="", description="Prompt to be used for the chat completion"
    )
    max_tokens: str = Field(
        default="", description="This sets the upper limit for the number of tokens the model can generate in response. It won't produce more than this limit. The maximum value is the context length minus the prompt length."
    )
    temperature: float = Field(
        default=1, description="This setting influences the variety in the model's responses. Lower values lead to more predictable and typical responses, while higher values encourage more diverse and less common responses. At 0, the model always gives the same response for a given input."
    )
    reasoning: bool = Field(
        default=False, description="Should reasoning be the part of the final answer."
    )
    system_prompt: str = Field(
        default="", description="System prompt to provide context or instructions to the model"
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "model": self.model,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "reasoning": self.reasoning,
            "system_prompt": self.system_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="openrouter/router",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "model"]

class OpenRouterChatCompletions(FALNode):
    """
    OpenRouter Chat Completions provides OpenAI-compatible interface for any LLM.
    llm, chat, openai-compatible, openrouter, chat-completions

    Use cases:
    - OpenAI-compatible LLM access
    - Drop-in replacement for OpenAI API
    - Multi-model chat completions
    - Standardized chat interface
    - Universal LLM chat API
    """


    async def process(self, context: ProcessingContext) -> Any:
        arguments = {
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="openrouter/router/openai/v1/chat/completions",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["messages", "model"]

class Qwen3Guard(FALNode):
    """
    Qwen 3 Guard provides content safety and moderation using Qwen's LLM.
    llm, safety, moderation, qwen, guard

    Use cases:
    - Content safety checking
    - Moderation of text content
    - Safety filtering for outputs
    - Content policy enforcement
    - Text safety analysis
    """

    class Label(Enum):
        """
        The classification label
        """
        SAFE = "Safe"
        UNSAFE = "Unsafe"
        CONTROVERSIAL = "Controversial"


    prompt: str = Field(
        default="", description="The input text to be classified"
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "prompt": self.prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-3-guard",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["text"]