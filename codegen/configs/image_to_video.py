"""
Configuration for image_to_video module.

This config file defines overrides and customizations for image-to-video nodes.
"""

from typing import Any


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
    "fal-ai/pixverse/v5.6/image-to-video": {
        "class_name": "PixverseV56ImageToVideo",
        "docstring": "Generate high-quality videos from images with Pixverse v5.6.",
        "tags": ["video", "generation", "pixverse", "v5.6", "image-to-video", "img2vid"],
        "use_cases": [
            "Animate photos into professional video clips",
            "Create dynamic product showcase videos",
            "Generate stylized video content from artwork",
            "Produce high-resolution social media animations",
            "Transform static images with various visual styles"
        ],
        "field_overrides": {
            "image_url": {
                "name": "image",  # Rename field
                "description": "The image to transform into a video"
            },
            "prompt": {
                "description": "Text prompt describing the desired video motion"
            },
            "resolution": {
                "description": "The resolution quality of the output video"
            },
            "duration": {
                "description": "The duration of the generated video in seconds"
            },
            "negative_prompt": {
                "description": "What to avoid in the generated video"
            },
            "style": {
                "description": "Optional visual style for the video"
            },
            "seed": {
                "description": "Optional seed for reproducible generation"
            },
            "generate_audio_switch": {
                "description": "Whether to generate audio for the video"
            },
            "thinking_type": {
                "description": "Thinking mode for video generation"
            }
        },
        "enum_overrides": {
            "Resolution": "PixverseV56Resolution",
            "Duration": "PixverseV56Duration",
            "Style": "PixverseV56Style",
            "ThinkingType": "PixverseV56ThinkingType"
        },
        "enum_value_overrides": {
            "Duration": {
                "VALUE_5": "FIVE_SECONDS",
                "VALUE_8": "EIGHT_SECONDS",
                "VALUE_10": "TEN_SECONDS"
            },
            "Resolution": {
                "360P": "RES_360P",
                "540P": "RES_540P",
                "720P": "RES_720P",
                "1080P": "RES_1080P"
            },
            "Style": {
                "3D_ANIMATION": "ANIMATION_3D"
            }
        },
        "basic_fields": ["image", "prompt", "resolution"]
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
