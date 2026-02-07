"""
Configuration for image_to_image module.

This config file defines overrides and customizations for image-to-image nodes.
"""

from typing import Any


# Shared enums used across multiple nodes
# Import ImageSizePreset from text_to_image module
SHARED_ENUMS = {}


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
    # FLUX Redux Family - Style Transfer
    "fal-ai/flux/schnell/redux": {
        "class_name": "FluxSchnellRedux",
        "docstring": "FLUX.1 [schnell] Redux enables rapid transformation of existing images with high-quality style transfers and modifications using the fast FLUX.1 schnell model.",
        "tags": ["image", "transformation", "style-transfer", "fast", "flux", "redux"],
        "use_cases": [
            "Transform images with artistic style transfers",
            "Apply quick modifications to photos",
            "Create image variations for rapid iteration",
            "Generate stylized versions of existing images",
            "Produce fast image transformations"
        ],
        "field_overrides": {
            "image": {
                "description": "The input image to transform"
            },
            "image_size": {
                "python_type": "ImageSizePreset",
                "default_value": "ImageSizePreset.LANDSCAPE_4_3",
                "description": "The size of the generated image"
            },
            "num_inference_steps": {
                "default_value": "4",
                "description": "The number of inference steps to perform (1-50)"
            },
            "seed": {
                "python_type": "int",
                "default_value": "-1",
                "description": "Seed for reproducible results. Use -1 for random"
            },
            "enable_safety_checker": {
                "description": "Enable safety checker to filter unsafe content"
            },
            "num_images": {
                "description": "The number of images to generate (1-4)"
            },
            "output_format": {
                "description": "Output format (jpeg or png)"
            },
            "acceleration": {
                "description": "Acceleration speed: 'none', 'regular', or 'high'"
            }
        },
        "enum_overrides": {
            "ImageSize": "ImageSizePreset"
        },
        "basic_fields": ["image", "image_size", "num_inference_steps"]
    },

    "fal-ai/flux/dev/redux": {
        "class_name": "FluxDevRedux",
        "docstring": "FLUX.1 [dev] Redux provides advanced image transformation capabilities with superior quality and more control over the style transfer process.",
        "tags": ["image", "transformation", "style-transfer", "development", "flux", "redux"],
        "use_cases": [
            "Transform images with advanced quality controls",
            "Create customized image variations with guidance",
            "Apply precise style modifications",
            "Generate high-quality artistic transformations",
            "Produce refined image edits with better prompt adherence"
        ],
        "field_overrides": {
            "image": {
                "description": "The input image to transform"
            },
            "image_size": {
                "python_type": "ImageSizePreset",
                "default_value": "ImageSizePreset.LANDSCAPE_4_3",
                "description": "The size of the generated image"
            },
            "num_inference_steps": {
                "default_value": "28",
                "description": "The number of inference steps to perform (1-50)"
            },
            "guidance_scale": {
                "description": "How strictly to follow the image structure (1-20)"
            },
            "seed": {
                "python_type": "int",
                "default_value": "-1",
                "description": "Seed for reproducible results. Use -1 for random"
            },
            "enable_safety_checker": {
                "description": "Enable safety checker to filter unsafe content"
            },
            "num_images": {
                "description": "The number of images to generate (1-4)"
            },
            "output_format": {
                "description": "Output format (jpeg or png)"
            }
        },
        "enum_overrides": {
            "ImageSize": "ImageSizePreset"
        },
        "basic_fields": ["image", "image_size", "guidance_scale"]
    },

    "fal-ai/flux-pro/v1/redux": {
        "class_name": "FluxProRedux",
        "docstring": "FLUX.1 Pro Redux delivers professional-grade image transformations with the highest quality and safety controls for commercial use.",
        "tags": ["image", "transformation", "style-transfer", "professional", "flux", "redux"],
        "use_cases": [
            "Create professional-quality image transformations",
            "Apply commercial-grade style transfers",
            "Generate high-fidelity image variations",
            "Produce brand-safe image modifications",
            "Transform images for production use"
        ],
        "field_overrides": {
            "image": {
                "description": "The input image to transform"
            },
            "image_size": {
                "python_type": "ImageSizePreset",
                "default_value": "ImageSizePreset.LANDSCAPE_4_3",
                "description": "The size of the generated image"
            },
            "num_inference_steps": {
                "default_value": "28",
                "description": "The number of inference steps to perform (1-50)"
            },
            "guidance_scale": {
                "description": "How strictly to follow the image structure (1-20)"
            },
            "seed": {
                "python_type": "int",
                "default_value": "-1",
                "description": "Seed for reproducible results. Use -1 for random"
            },
            "safety_tolerance": {
                "description": "Safety tolerance level (1-6, higher is stricter)"
            },
            "num_images": {
                "description": "The number of images to generate (1-4)"
            },
            "output_format": {
                "description": "Output format (jpeg or png)"
            }
        },
        "enum_overrides": {
            "ImageSize": "ImageSizePreset"
        },
        "basic_fields": ["image", "image_size", "guidance_scale"]
    },

    # Ideogram Family - Editing
    "fal-ai/ideogram/v2/edit": {
        "class_name": "IdeogramV2Edit",
        "docstring": "Transform existing images with Ideogram V2's editing capabilities. Modify, adjust, and refine images while maintaining high fidelity with precise prompt and mask control.",
        "tags": ["image", "editing", "inpainting", "mask", "ideogram", "transformation"],
        "use_cases": [
            "Edit specific parts of images with precision",
            "Create targeted image modifications using masks",
            "Refine and enhance image details",
            "Generate contextual image edits",
            "Replace or modify masked regions"
        ],
        "field_overrides": {
            "prompt": {
                "description": "The prompt to fill the masked part of the image"
            },
            "image": {
                "description": "The image to edit"
            },
            "mask": {
                "description": "The mask defining areas to edit (white = edit, black = keep)"
            },
            "style": {
                "description": "Style of generated image (auto, general, realistic, design, render_3D, anime)"
            },
            "expand_prompt": {
                "description": "Whether to expand the prompt with MagicPrompt functionality"
            },
            "seed": {
                "python_type": "int",
                "default_value": "-1",
                "description": "Seed for reproducible results. Use -1 for random"
            }
        },
        "basic_fields": ["prompt", "image", "mask"]
    },

    "fal-ai/ideogram/v2/remix": {
        "class_name": "IdeogramV2Remix",
        "docstring": "Reimagine existing images with Ideogram V2's remix feature. Create variations and adaptations while preserving core elements through prompt guidance and strength control.",
        "tags": ["image", "remix", "variation", "creativity", "ideogram", "adaptation"],
        "use_cases": [
            "Create artistic variations of images",
            "Generate style-transferred versions",
            "Produce creative image adaptations",
            "Transform images while preserving key elements",
            "Generate alternative interpretations"
        ],
        "field_overrides": {
            "prompt": {
                "description": "The prompt to remix the image with"
            },
            "image": {
                "description": "The image to remix"
            },
            "aspect_ratio": {
                "description": "The aspect ratio of the generated image"
            },
            "strength": {
                "description": "Strength of the input image in the remix (0-1, higher = more variation)"
            },
            "expand_prompt": {
                "description": "Whether to expand the prompt with MagicPrompt functionality"
            },
            "style": {
                "description": "Style of generated image (auto, general, realistic, design, render_3D, anime)"
            },
            "seed": {
                "python_type": "int",
                "default_value": "-1",
                "description": "Seed for reproducible results. Use -1 for random"
            }
        },
        "basic_fields": ["prompt", "image", "strength"]
    },

    "fal-ai/ideogram/v3/edit": {
        "class_name": "IdeogramV3Edit",
        "docstring": "Transform images with Ideogram V3's enhanced editing capabilities. Latest generation editing with improved quality, control, and style consistency.",
        "tags": ["image", "editing", "inpainting", "mask", "ideogram", "v3"],
        "use_cases": [
            "Edit images with the latest Ideogram technology",
            "Apply high-fidelity masked edits",
            "Generate professional image modifications",
            "Create precise content-aware fills",
            "Refine image details with advanced controls"
        ],
        "field_overrides": {
            "prompt": {
                "description": "The prompt to fill the masked part of the image"
            },
            "image": {
                "description": "The image to edit"
            },
            "mask": {
                "description": "The mask defining areas to edit (white = edit, black = keep)"
            },
            "style": {
                "description": "Style of generated image"
            },
            "seed": {
                "python_type": "int",
                "default_value": "-1",
                "description": "Seed for reproducible results. Use -1 for random"
            }
        },
        "basic_fields": ["prompt", "image", "mask"]
    },

    # FLUX Pro Family - Advanced Controls
    "fal-ai/flux-pro/v1/fill": {
        "class_name": "FluxProFill",
        "docstring": "FLUX.1 Pro Fill provides professional inpainting and outpainting capabilities. Generate or modify image content within masked regions with precise prompt control.",
        "tags": ["image", "inpainting", "outpainting", "fill", "flux", "professional"],
        "use_cases": [
            "Fill masked regions with new content",
            "Extend images beyond their boundaries (outpainting)",
            "Remove unwanted objects and fill gaps",
            "Generate content-aware image expansions",
            "Create seamless image modifications"
        ],
        "field_overrides": {
            "prompt": {
                "description": "The prompt describing what to generate in the masked area"
            },
            "image": {
                "description": "The image to fill"
            },
            "mask": {
                "description": "The mask defining areas to fill (white = fill, black = keep)"
            },
            "image_size": {
                "python_type": "ImageSizePreset",
                "default_value": "ImageSizePreset.LANDSCAPE_4_3",
                "description": "The size of the generated image"
            },
            "num_inference_steps": {
                "description": "The number of inference steps to perform"
            },
            "guidance_scale": {
                "description": "How strictly to follow the prompt"
            },
            "seed": {
                "python_type": "int",
                "default_value": "-1",
                "description": "Seed for reproducible results. Use -1 for random"
            },
            "safety_tolerance": {
                "description": "Safety tolerance level (1-6, higher is stricter)"
            }
        },
        "enum_overrides": {
            "ImageSize": "ImageSizePreset"
        },
        "basic_fields": ["prompt", "image", "mask"]
    },

    "fal-ai/flux-pro/v1/canny": {
        "class_name": "FluxProCanny",
        "docstring": "FLUX.1 Pro with Canny edge detection control. Generate images guided by edge maps for precise structural control while maintaining FLUX's quality.",
        "tags": ["image", "controlnet", "canny", "edges", "flux", "professional"],
        "use_cases": [
            "Generate images following edge structures",
            "Transform images while preserving edges",
            "Create controlled variations with edge guidance",
            "Apply style transfers with structural constraints",
            "Generate content from edge maps"
        ],
        "field_overrides": {
            "prompt": {
                "description": "The text prompt describing the desired output"
            },
            "image": {
                "description": "The control image (edges will be detected automatically)"
            },
            "image_size": {
                "python_type": "ImageSizePreset",
                "default_value": "ImageSizePreset.LANDSCAPE_4_3",
                "description": "The size of the generated image"
            },
            "num_inference_steps": {
                "description": "The number of inference steps to perform"
            },
            "guidance_scale": {
                "description": "How strictly to follow the prompt"
            },
            "seed": {
                "python_type": "int",
                "default_value": "-1",
                "description": "Seed for reproducible results. Use -1 for random"
            },
            "control_strength": {
                "description": "How strongly to follow the edge structure (0-1)"
            }
        },
        "enum_overrides": {
            "ImageSize": "ImageSizePreset"
        },
        "basic_fields": ["prompt", "image", "control_strength"]
    },

    "fal-ai/flux-pro/v1/depth": {
        "class_name": "FluxProDepth",
        "docstring": "FLUX.1 Pro with depth map control. Generate images guided by depth information for precise 3D structure control while maintaining FLUX's quality.",
        "tags": ["image", "controlnet", "depth", "3d", "flux", "professional"],
        "use_cases": [
            "Generate images following depth structures",
            "Transform images while preserving 3D composition",
            "Create controlled variations with depth guidance",
            "Apply style transfers with spatial constraints",
            "Generate content from depth maps"
        ],
        "field_overrides": {
            "prompt": {
                "description": "The text prompt describing the desired output"
            },
            "image": {
                "description": "The control image (depth will be estimated automatically)"
            },
            "image_size": {
                "python_type": "ImageSizePreset",
                "default_value": "ImageSizePreset.LANDSCAPE_4_3",
                "description": "The size of the generated image"
            },
            "num_inference_steps": {
                "description": "The number of inference steps to perform"
            },
            "guidance_scale": {
                "description": "How strictly to follow the prompt"
            },
            "seed": {
                "python_type": "int",
                "default_value": "-1",
                "description": "Seed for reproducible results. Use -1 for random"
            },
            "control_strength": {
                "description": "How strongly to follow the depth structure (0-1)"
            }
        },
        "enum_overrides": {
            "ImageSize": "ImageSizePreset"
        },
        "basic_fields": ["prompt", "image", "control_strength"]
    },

    # Bria Family - Professional Editing Tools
    "bria/eraser": {
        "class_name": "BriaEraser",
        "docstring": "Bria Eraser removes unwanted objects from images using intelligent inpainting. Seamlessly fill removed areas with contextually appropriate content.",
        "tags": ["image", "eraser", "removal", "inpainting", "bria", "cleanup"],
        "use_cases": [
            "Remove unwanted objects from photos",
            "Clean up image backgrounds",
            "Erase text or watermarks",
            "Delete distracting elements",
            "Create clean product shots"
        ],
        "field_overrides": {
            "image": {
                "description": "The image containing objects to remove"
            },
            "mask": {
                "description": "The mask defining areas to erase (white = erase, black = keep)"
            }
        },
        "basic_fields": ["image", "mask"]
    },

    "bria/replace-background": {
        "class_name": "BriaBackgroundReplace",
        "docstring": "Bria Background Replace swaps image backgrounds with new content. Intelligently separates subjects and generates contextually appropriate backgrounds.",
        "tags": ["image", "background", "replacement", "segmentation", "bria"],
        "use_cases": [
            "Replace photo backgrounds with custom scenes",
            "Create product shots with various backgrounds",
            "Change image context while preserving subject",
            "Generate professional portraits with studio backgrounds",
            "Create marketing materials with branded backgrounds"
        ],
        "field_overrides": {
            "image": {
                "description": "The image whose background to replace"
            },
            "prompt": {
                "description": "Description of the new background to generate"
            }
        },
        "basic_fields": ["image", "prompt"]
    },

    # Enhancement/Upscaling
    "fal-ai/clarity-upscaler": {
        "class_name": "ClarityUpscaler",
        "docstring": "Clarity Upscaler increases image resolution using AI-powered super-resolution. Enhance image quality, sharpness, and detail up to 4x scale.",
        "tags": ["image", "upscaling", "enhancement", "super-resolution", "clarity"],
        "use_cases": [
            "Increase image resolution for printing",
            "Improve clarity of low-quality images",
            "Enhance textures and fine details",
            "Prepare images for large displays",
            "Restore detail in compressed images"
        ],
        "field_overrides": {
            "image": {
                "description": "Input image to upscale"
            },
            "scale": {
                "default_value": "2",
                "description": "Upscaling factor (1-4x)"
            }
        },
        "basic_fields": ["image", "scale"]
    },

    # Alternative Model Families
    "fal-ai/recraft/v3/image-to-image": {
        "class_name": "RecraftV3ImageToImage",
        "docstring": "Recraft V3 transforms images with advanced style control and quality preservation. Professional-grade image-to-image generation with fine-tuned artistic control.",
        "tags": ["image", "transformation", "recraft", "style", "professional"],
        "use_cases": [
            "Transform images with precise style control",
            "Create high-quality image variations",
            "Apply artistic modifications with consistency",
            "Generate professional design alternatives",
            "Produce style-coherent image transformations"
        ],
        "field_overrides": {
            "prompt": {
                "description": "The text prompt describing the desired transformation"
            },
            "image": {
                "description": "The input image to transform"
            },
            "style": {
                "description": "The artistic style to apply"
            },
            "seed": {
                "python_type": "int",
                "default_value": "-1",
                "description": "Seed for reproducible results. Use -1 for random"
            }
        },
        "basic_fields": ["prompt", "image", "style"]
    },

    "fal-ai/kolors/image-to-image": {
        "class_name": "KolorsImageToImage",
        "docstring": "Kolors transforms images using an advanced diffusion model. High-quality image-to-image generation with natural color preservation and detail retention.",
        "tags": ["image", "transformation", "kolors", "diffusion", "quality"],
        "use_cases": [
            "Transform images with natural color handling",
            "Create variations with preserved color harmony",
            "Apply modifications with detail retention",
            "Generate style transfers with color consistency",
            "Produce high-fidelity image transformations"
        ],
        "field_overrides": {
            "prompt": {
                "description": "The text prompt describing the desired transformation"
            },
            "image": {
                "description": "The input image to transform"
            },
            "strength": {
                "description": "Strength of the transformation (0-1, higher = more change)"
            },
            "seed": {
                "python_type": "int",
                "default_value": "-1",
                "description": "Seed for reproducible results. Use -1 for random"
            }
        },
        "basic_fields": ["prompt", "image", "strength"]
    },

    # Specialized Tools
    "fal-ai/birefnet": {
        "class_name": "BiRefNet",
        "docstring": "BiRefNet (Bilateral Reference Network) performs high-quality background removal with precise edge detection and detail preservation.",
        "tags": ["image", "background-removal", "segmentation", "birefnet", "mask"],
        "use_cases": [
            "Remove backgrounds from product photos",
            "Create transparent PNGs from images",
            "Extract subjects for compositing",
            "Generate clean cutouts for design work",
            "Prepare images for background replacement"
        ],
        "field_overrides": {
            "image": {
                "description": "The image to remove background from"
            }
        },
        "basic_fields": ["image"]
    },

    "fal-ai/codeformer": {
        "class_name": "CodeFormer",
        "docstring": "CodeFormer restores and enhances face quality in images. Advanced face restoration with fidelity control for natural-looking results.",
        "tags": ["image", "face-restoration", "enhancement", "codeformer", "quality"],
        "use_cases": [
            "Restore quality in degraded face photos",
            "Enhance facial details in low-quality images",
            "Improve portrait quality for professional use",
            "Fix compressed or damaged face images",
            "Enhance facial features while maintaining identity"
        ],
        "field_overrides": {
            "image": {
                "description": "The image containing faces to restore"
            },
            "fidelity": {
                "description": "Fidelity level (0-1, higher = more faithful to input)"
            }
        },
        "basic_fields": ["image", "fidelity"]
    },
}


def get_config(endpoint_id: str) -> dict[str, Any]:
    """
    Get configuration for an endpoint.
    
    Args:
        endpoint_id: FAL endpoint ID
        
    Returns:
        Configuration dictionary, or empty dict if not found
    """
    return CONFIGS.get(endpoint_id, {})
