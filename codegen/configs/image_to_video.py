"""
Configuration for image_to_video module.

This config file defines overrides and customizations for image-to-video nodes.
"""

from typing import Any


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
    "fal-ai/haiper-video-v2/image-to-video": {
        "class_name": "HaiperImageToVideo",
        "docstring": "Transform images into hyper-realistic videos with Haiper 2.0. Experience industry-leading resolution, fluid motion, and rapid generation for stunning AI videos.",
        "tags": ["video", "generation", "hyper-realistic", "motion", "animation", "image-to-video", "img2vid"],
        "use_cases": [
            "Create cinematic animations",
            "Generate dynamic video content",
            "Transform static images into motion",
            "Produce high-resolution videos",
            "Create visual effects"
        ],
        "field_overrides": {
            "image": {
                "python_type": "ImageRef",
                "default_value": "ImageRef()",
                "description": "The image to transform into a video"
            },
            "prompt": {
                "description": "A description of the desired video motion and style"
            },
            "duration": {
                "description": "The duration of the generated video in seconds"
            }
        }
    },
    
    "fal-ai/luma-dream-machine/image-to-video": {
        "class_name": "LumaDreamMachine",
        "docstring": "Generate video clips from your images using Luma Dream Machine v1.5. Supports various aspect ratios and optional end-frame blending.",
        "tags": ["video", "generation", "animation", "blending", "aspect-ratio", "img2vid", "image-to-video"],
        "use_cases": [
            "Create seamless video loops",
            "Generate video transitions",
            "Transform images into animations",
            "Create motion graphics",
            "Produce video content"
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
