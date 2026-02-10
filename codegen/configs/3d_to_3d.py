"""
Configuration for 3d-to-3d module.

This config file defines overrides and customizations for 3d-to-3d nodes.
"""

from typing import Any


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
"fal-ai/ultrashape": {
        "class_name": "Ultrashape",
        "docstring": "Ultrashape",
        "tags": ["3d_to_3d"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/sam-3/3d-align": {
        "class_name": "Sam33DAlign",
        "docstring": "Sam 3",
        "tags": ["3d_to_3d"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/meshy/v5/retexture": {
        "class_name": "MeshyV5Retexture",
        "docstring": "Meshy-5 retexture applies new, high-quality textures to existing 3D models using either text prompts or reference images. It supports PBR material generation for realistic, production-ready results.",
        "tags": ["3d", "editing", "transformation", "modeling"],
        "use_cases": [
            "3D model editing and refinement",
            "Mesh optimization",
            "Texture application",
            "3D format conversion",
            "Model retopology",
        ],
    },
    "fal-ai/meshy/v5/remesh": {
        "class_name": "MeshyV5Remesh",
        "docstring": "Meshy-5 remesh allows you to remesh and export existing 3D models into various formats",
        "tags": ["3d", "editing", "transformation", "modeling"],
        "use_cases": [
            "3D model editing and refinement",
            "Mesh optimization",
            "Texture application",
            "3D format conversion",
            "Model retopology",
        ],
    },
    "fal-ai/hunyuan-part": {
        "class_name": "HunyuanPart",
        "docstring": "Use the capabilities of hunyuan part to generate point clouds from your 3D files.",
        "tags": ["3d", "editing", "transformation", "modeling"],
        "use_cases": [
            "3D model editing and refinement",
            "Mesh optimization",
            "Texture application",
            "3D format conversion",
            "Model retopology",
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
