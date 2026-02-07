"""
Configuration for training module.

This config file defines overrides and customizations for training nodes.
"""

from typing import Any


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
"fal-ai/z-image-base-trainer": {
        "class_name": "ZImageBaseTrainer",
        "docstring": "Z-Image Trainer",
        "tags": ["training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/z-image-turbo-trainer-v2": {
        "class_name": "ZImageTurboTrainerV2",
        "docstring": "Z Image Turbo Trainer V2",
        "tags": ["training", "fine-tuning", "lora", "model-training", "fast"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/flux-2-klein-9b-base-trainer/edit": {
        "class_name": "Flux2Klein9BBaseTrainerEdit",
        "docstring": "Flux 2 Klein 9B Base Trainer",
        "tags": ["flux", "training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/flux-2-klein-9b-base-trainer": {
        "class_name": "Flux2Klein9BBaseTrainer",
        "docstring": "Flux 2 Klein 9B Base Trainer",
        "tags": ["flux", "training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/flux-2-klein-4b-base-trainer": {
        "class_name": "Flux2Klein4BBaseTrainer",
        "docstring": "Flux 2 Klein 4B Base Trainer",
        "tags": ["flux", "training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/flux-2-klein-4b-base-trainer/edit": {
        "class_name": "Flux2Klein4BBaseTrainerEdit",
        "docstring": "Flux 2 Klein 4B Base Trainer",
        "tags": ["flux", "training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/qwen-image-2512-trainer-v2": {
        "class_name": "QwenImage2512TrainerV2",
        "docstring": "Qwen Image 2512 Trainer V2",
        "tags": ["training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/flux-2-trainer-v2/edit": {
        "class_name": "Flux2TrainerV2Edit",
        "docstring": "Flux 2 Trainer V2",
        "tags": ["flux", "training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/flux-2-trainer-v2": {
        "class_name": "Flux2TrainerV2",
        "docstring": "Flux 2 Trainer V2",
        "tags": ["flux", "training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/ltx2-v2v-trainer": {
        "class_name": "Ltx2V2VTrainer",
        "docstring": "LTX-2 Video to Video Trainer",
        "tags": ["training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/ltx2-video-trainer": {
        "class_name": "Ltx2VideoTrainer",
        "docstring": "LTX-2 Video Trainer",
        "tags": ["training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/qwen-image-2512-trainer": {
        "class_name": "QwenImage2512Trainer",
        "docstring": "Qwen Image 2512 Trainer",
        "tags": ["training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/qwen-image-edit-2511-trainer": {
        "class_name": "QwenImageEdit2511Trainer",
        "docstring": "Qwen Image Edit 2511 Trainer",
        "tags": ["training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/qwen-image-layered-trainer": {
        "class_name": "QwenImageLayeredTrainer",
        "docstring": "Qwen Image Layered Trainer",
        "tags": ["training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/qwen-image-edit-2509-trainer": {
        "class_name": "QwenImageEdit2509Trainer",
        "docstring": "Qwen Image Edit 2509 Trainer",
        "tags": ["training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
}


def get_config(endpoint_id: str) -> dict[str, Any]:
    """
    Get configuration for an endpoint.
    
    Args:
        endpoint_id: FAL endpoint ID
        
    Returns:
        Configuration dictionary
    """
    return CONFIGS.get(endpoint_id, {})
