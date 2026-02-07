"""
Configuration for llm module.

This config file defines overrides and customizations for LLM nodes.
"""

from typing import Any


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
    "openrouter/router": {
        "class_name": "OpenRouter",
        "docstring": "OpenRouter provides unified access to any LLM (Large Language Model) through a single API.",
        "tags": ["llm", "chat", "openrouter", "multimodel", "language-model"],
        "use_cases": [
            "Run any LLM through unified interface",
            "Switch between models seamlessly",
            "Access multiple LLM providers",
            "Flexible model selection",
            "Unified LLM API access"
        ],
        "basic_fields": ["prompt", "model"]
    },
    
    "openrouter/router/openai/v1/chat/completions": {
        "class_name": "OpenRouterChatCompletions",
        "docstring": "OpenRouter Chat Completions provides OpenAI-compatible interface for any LLM.",
        "tags": ["llm", "chat", "openai-compatible", "openrouter", "chat-completions"],
        "use_cases": [
            "OpenAI-compatible LLM access",
            "Drop-in replacement for OpenAI API",
            "Multi-model chat completions",
            "Standardized chat interface",
            "Universal LLM chat API"
        ],
        "basic_fields": ["messages", "model"]
    },
    
    "fal-ai/qwen-3-guard": {
        "class_name": "Qwen3Guard",
        "docstring": "Qwen 3 Guard provides content safety and moderation using Qwen's LLM.",
        "tags": ["llm", "safety", "moderation", "qwen", "guard"],
        "use_cases": [
            "Content safety checking",
            "Moderation of text content",
            "Safety filtering for outputs",
            "Content policy enforcement",
            "Text safety analysis"
        ],
        "basic_fields": ["text"]
    },
}


def get_config(endpoint_id: str) -> dict[str, Any]:
    """Get config for an endpoint."""
    return CONFIGS.get(endpoint_id, {})
