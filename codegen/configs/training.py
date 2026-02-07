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
    "fal-ai/z-image-trainer": {
        "class_name": "ZImageTrainer",
        "docstring": "Train LoRAs on Z-Image Turbo, a super fast text-to-image model of 6B parameters developed by Tongyi-MAI.",
        "tags": ["training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Custom model fine-tuning",
            "LoRA training for personalization",
            "Style-specific model training",
            "Brand-specific image generation",
            "Specialized domain adaptation",
        ],
    },
    "fal-ai/flux-2-trainer/edit": {
        "class_name": "Flux2TrainerEdit",
        "docstring": "Fine-tune FLUX.2 [dev] from Black Forest Labs with custom datasets. Create specialized LoRA adaptations for specific editing tasks.",
        "tags": ["flux", "training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Custom model fine-tuning",
            "LoRA training for personalization",
            "Style-specific model training",
            "Brand-specific image generation",
            "Specialized domain adaptation",
        ],
    },
    "fal-ai/flux-2-trainer": {
        "class_name": "Flux2Trainer",
        "docstring": "Fine-tune FLUX.2 [dev] from Black Forest Labs with custom datasets. Create specialized LoRA adaptations for specific styles and domains.",
        "tags": ["flux", "training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Custom model fine-tuning",
            "LoRA training for personalization",
            "Style-specific model training",
            "Brand-specific image generation",
            "Specialized domain adaptation",
        ],
    },
    "fal-ai/qwen-image-edit-plus-trainer": {
        "class_name": "QwenImageEditPlusTrainer",
        "docstring": "LoRA trainer for Qwen Image Edit Plus",
        "tags": ["training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Custom model fine-tuning",
            "LoRA training for personalization",
            "Style-specific model training",
            "Brand-specific image generation",
            "Specialized domain adaptation",
        ],
    },
    "fal-ai/qwen-image-edit-trainer": {
        "class_name": "QwenImageEditTrainer",
        "docstring": "LoRA trainer for Qwen Image Edit",
        "tags": ["training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Custom model fine-tuning",
            "LoRA training for personalization",
            "Style-specific model training",
            "Brand-specific image generation",
            "Specialized domain adaptation",
        ],
    },
    "fal-ai/qwen-image-trainer": {
        "class_name": "QwenImageTrainer",
        "docstring": "Qwen Image LoRA training",
        "tags": ["training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Custom model fine-tuning",
            "LoRA training for personalization",
            "Style-specific model training",
            "Brand-specific image generation",
            "Specialized domain adaptation",
        ],
    },
    "fal-ai/wan-22-image-trainer": {
        "class_name": "Wan22ImageTrainer",
        "docstring": "Wan 2.2 text to image LoRA trainer. Fine-tune Wan 2.2 for subjects and styles with unprecedented detail.",
        "tags": ["training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Custom model fine-tuning",
            "LoRA training for personalization",
            "Style-specific model training",
            "Brand-specific image generation",
            "Specialized domain adaptation",
        ],
    },
    "fal-ai/wan-trainer/t2v": {
        "class_name": "WanTrainerT2v",
        "docstring": "Train custom LoRAs for Wan-2.1 T2V 1.3B",
        "tags": ["training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Custom model fine-tuning",
            "LoRA training for personalization",
            "Style-specific model training",
            "Brand-specific image generation",
            "Specialized domain adaptation",
        ],
    },
    "fal-ai/wan-trainer/t2v-14b": {
        "class_name": "WanTrainerT2v14b",
        "docstring": "Train custom LoRAs for Wan-2.1 T2V 14B",
        "tags": ["training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Custom model fine-tuning",
            "LoRA training for personalization",
            "Style-specific model training",
            "Brand-specific image generation",
            "Specialized domain adaptation",
        ],
    },
    "fal-ai/wan-trainer/i2v-720p": {
        "class_name": "WanTrainerI2v720p",
        "docstring": "Train custom LoRAs for Wan-2.1 I2V 720P",
        "tags": ["training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Custom model fine-tuning",
            "LoRA training for personalization",
            "Style-specific model training",
            "Brand-specific image generation",
            "Specialized domain adaptation",
        ],
    },
    "fal-ai/wan-trainer/flf2v-720p": {
        "class_name": "WanTrainerFlf2v720p",
        "docstring": "Train custom LoRAs for Wan-2.1 FLF2V 720P",
        "tags": ["training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Custom model fine-tuning",
            "LoRA training for personalization",
            "Style-specific model training",
            "Brand-specific image generation",
            "Specialized domain adaptation",
        ],
    },
    "fal-ai/ltx-video-trainer": {
        "class_name": "LtxVideoTrainer",
        "docstring": "Train LTX Video 0.9.7 for custom styles and effects.",
        "tags": ["training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Custom model fine-tuning",
            "LoRA training for personalization",
            "Style-specific model training",
            "Brand-specific image generation",
            "Specialized domain adaptation",
        ],
    },
    "fal-ai/recraft/v3/create-style": {
        "class_name": "RecraftV3CreateStyle",
        "docstring": "Recraft V3 Create Style is capable of creating unique styles for Recraft V3 based on your images.",
        "tags": ["training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Custom model fine-tuning",
            "LoRA training for personalization",
            "Style-specific model training",
            "Brand-specific image generation",
            "Specialized domain adaptation",
        ],
    },
    "fal-ai/turbo-flux-trainer": {
        "class_name": "TurboFluxTrainer",
        "docstring": "A blazing fast FLUX dev LoRA trainer for subjects and styles.",
        "tags": ["flux", "training", "fine-tuning", "lora", "model-training", "fast"],
        "use_cases": [
            "Custom model fine-tuning",
            "LoRA training for personalization",
            "Style-specific model training",
            "Brand-specific image generation",
            "Specialized domain adaptation",
        ],
    },
    "fal-ai/wan-trainer": {
        "class_name": "WanTrainer",
        "docstring": "Train custom LoRAs for Wan-2.1 I2V 480P",
        "tags": ["training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Custom model fine-tuning",
            "LoRA training for personalization",
            "Style-specific model training",
            "Brand-specific image generation",
            "Specialized domain adaptation",
        ],
    },
    "fal-ai/hunyuan-video-lora-training": {
        "class_name": "HunyuanVideoLoraTraining",
        "docstring": "Train Hunyuan Video lora on people, objects, characters and more!",
        "tags": ["training", "fine-tuning", "lora", "model-training"],
        "use_cases": [
            "Custom model fine-tuning",
            "LoRA training for personalization",
            "Style-specific model training",
            "Brand-specific image generation",
            "Specialized domain adaptation",
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
