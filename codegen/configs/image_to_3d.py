"""
Configuration for image-to-3d module.

This config file defines overrides and customizations for image-to-3d nodes.
"""

from typing import Any


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
"fal-ai/trellis-2": {
        "class_name": "Trellis2",
        "docstring": "Trellis 2",
        "tags": ["3d", "generation", "image-to-3d", "modeling"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/hunyuan3d-v3/sketch-to-3d": {
        "class_name": "Hunyuan3DV3SketchTo3D",
        "docstring": "Hunyuan3d V3",
        "tags": ["3d", "generation", "image-to-3d", "modeling"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/hunyuan3d-v3/image-to-3d": {
        "class_name": "Hunyuan3DV3ImageTo3D",
        "docstring": "Hunyuan3d V3",
        "tags": ["3d", "generation", "image-to-3d", "modeling"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/sam-3/3d-body": {
        "class_name": "Sam33DBody",
        "docstring": "Sam 3",
        "tags": ["3d", "generation", "image-to-3d", "modeling"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/sam-3/3d-objects": {
        "class_name": "Sam33DObjects",
        "docstring": "Sam 3",
        "tags": ["3d", "generation", "image-to-3d", "modeling"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/omnipart": {
        "class_name": "Omnipart",
        "docstring": "Omnipart",
        "tags": ["3d", "generation", "image-to-3d", "modeling"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/bytedance/seed3d/image-to-3d": {
        "class_name": "BytedanceSeed3DImageTo3D",
        "docstring": "Bytedance",
        "tags": ["3d", "generation", "image-to-3d", "modeling"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/meshy/v5/multi-image-to-3d": {
        "class_name": "MeshyV5MultiImageTo3D",
        "docstring": "Meshy 5 Multi",
        "tags": ["3d", "generation", "image-to-3d", "modeling"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/meshy/v6-preview/image-to-3d": {
        "class_name": "MeshyV6PreviewImageTo3D",
        "docstring": "Meshy 6 Preview",
        "tags": ["3d", "generation", "image-to-3d", "modeling"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/hyper3d/rodin/v2": {
        "class_name": "Hyper3DRodinV2",
        "docstring": "Hyper3d",
        "tags": ["3d", "generation", "image-to-3d", "modeling"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/pshuman": {
        "class_name": "Pshuman",
        "docstring": "Pshuman",
        "tags": ["3d", "generation", "image-to-3d", "modeling"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/hunyuan_world/image-to-world": {
        "class_name": "Hunyuan_WorldImageToWorld",
        "docstring": "Hunyuan World",
        "tags": ["3d", "generation", "image-to-3d", "modeling"],
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
