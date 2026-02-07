"""
Configuration for text_to_3d module.

This config file defines overrides and customizations for text-to-3d nodes.
"""

from typing import Any


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
    "fal-ai/hunyuan-motion/fast": {
        "class_name": "HunyuanMotionFast",
        "docstring": "Generate 3D human motions via text-to-generation interface of Hunyuan Motion!",
        "tags": ["3d", "generation", "text-to-3d", "modeling", "fast"],
        "use_cases": [
            "3D model generation from text",
            "Concept visualization",
            "Game asset creation",
            "Architectural prototyping",
            "Product design visualization",
        ],
    },
    "fal-ai/hunyuan-motion": {
        "class_name": "HunyuanMotion",
        "docstring": "Generate 3D human motions via text-to-generation interface of Hunyuan Motion!",
        "tags": ["3d", "generation", "text-to-3d", "modeling"],
        "use_cases": [
            "3D model generation from text",
            "Concept visualization",
            "Game asset creation",
            "Architectural prototyping",
            "Product design visualization",
        ],
    },
    "fal-ai/hunyuan3d-v3/text-to-3d": {
        "class_name": "Hunyuan3dV3TextTo3d",
        "docstring": "Turn simple sketches into detailed, fully-textured 3D models. Instantly convert your concept designs into formats ready for Unity, Unreal, and Blender.",
        "tags": ["3d", "generation", "text-to-3d", "modeling"],
        "use_cases": [
            "3D model generation from text",
            "Concept visualization",
            "Game asset creation",
            "Architectural prototyping",
            "Product design visualization",
        ],
    },
    "fal-ai/meshy/v6-preview/text-to-3d": {
        "class_name": "MeshyV6PreviewTextTo3d",
        "docstring": "Meshy-6-Preview is the latest model from Meshy. It generates realistic and production ready 3D models.",
        "tags": ["3d", "generation", "text-to-3d", "modeling"],
        "use_cases": [
            "3D model generation from text",
            "Concept visualization",
            "Game asset creation",
            "Architectural prototyping",
            "Product design visualization",
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
