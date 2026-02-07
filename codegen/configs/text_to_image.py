"""
Configuration for text_to_image module.

This config file defines overrides and customizations for text-to-image nodes.
"""

from typing import Any


# Shared enums used across multiple nodes
SHARED_ENUMS = {
    "ImageSizePreset": {
        "values": [
            ("SQUARE_HD", "square_hd"),
            ("SQUARE", "square"),
            ("PORTRAIT_4_3", "portrait_4_3"),
            ("PORTRAIT_16_9", "portrait_16_9"),
            ("LANDSCAPE_4_3", "landscape_4_3"),
            ("LANDSCAPE_16_9", "landscape_16_9"),
        ],
        "description": "Preset sizes for image generation"
    }
}


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
    "fal-ai/flux/dev": {
        "class_name": "FluxDev",
        "docstring": "FLUX.1 [dev] is a powerful open-weight text-to-image model with 12 billion parameters. Optimized for prompt following and visual quality.",
        "tags": ["image", "generation", "flux", "text-to-image", "txt2img"],
        "use_cases": [
            "Generate high-quality images from text prompts",
            "Create detailed illustrations with precise control",
            "Produce professional artwork and designs",
            "Generate multiple variations from one prompt",
            "Create safe-for-work content with built-in safety checker"
        ],
        "field_overrides": {
            "prompt": {
                "description": "The prompt to generate an image from"
            },
            "image_size": {
                "python_type": "ImageSizePreset",
                "default_value": "ImageSizePreset.LANDSCAPE_4_3",
                "description": "Size preset for the generated image"
            },
            "num_inference_steps": {
                "description": "Number of denoising steps. More steps typically improve quality"
            },
            "guidance_scale": {
                "description": "How strictly to follow the prompt. Higher values are more literal"
            },
            "num_images": {
                "description": "Number of images to generate"
            },
            "seed": {
                "python_type": "int",
                "default_value": "-1",
                "description": "Seed for reproducible results. Use -1 for random"
            },
            "enable_safety_checker": {
                "description": "Enable safety checker to filter unsafe content"
            }
        },
        "enum_overrides": {
            "ImageSize": "ImageSizePreset"
        },
        "basic_fields": ["prompt", "image_size", "num_inference_steps"]
    },
    
    "fal-ai/flux/schnell": {
        "class_name": "FluxSchnell",
        "docstring": "FLUX.1 [schnell] is a fast distilled version of FLUX.1 optimized for speed. Can generate high-quality images in 1-4 steps.",
        "tags": ["image", "generation", "flux", "fast", "text-to-image", "txt2img"],
        "use_cases": [
            "Generate images quickly for rapid iteration",
            "Create concept art with minimal latency",
            "Produce preview images before final generation",
            "Generate multiple variations efficiently",
            "Real-time image generation applications"
        ],
        "field_overrides": {
            "prompt": {
                "description": "The prompt to generate an image from"
            },
            "image_size": {
                "python_type": "ImageSizePreset",
                "default_value": "ImageSizePreset.LANDSCAPE_4_3",
                "description": "Size preset for the generated image"
            },
            "num_inference_steps": {
                "description": "Number of denoising steps (1-4 recommended for schnell)"
            },
            "num_images": {
                "description": "Number of images to generate"
            },
            "seed": {
                "python_type": "int",
                "default_value": "-1",
                "description": "Seed for reproducible results. Use -1 for random"
            },
            "enable_safety_checker": {
                "description": "Enable safety checker to filter unsafe content"
            }
        },
        "enum_overrides": {
            "ImageSize": "ImageSizePreset"
        },
        "basic_fields": ["prompt", "image_size", "num_inference_steps"]
    },
    
    "fal-ai/flux-pro/v1.1": {
        "class_name": "FluxV1Pro",
        "docstring": "FLUX.1 Pro is a state-of-the-art image generation model with superior prompt following and image quality.",
        "tags": ["image", "generation", "flux", "pro", "text-to-image", "txt2img"],
        "use_cases": [
            "Generate professional-grade images for commercial use",
            "Create highly detailed artwork with complex prompts",
            "Produce marketing materials and brand assets",
            "Generate photorealistic images",
            "Create custom visual content with precise control"
        ],
        "field_overrides": {
            "prompt": {
                "description": "The prompt to generate an image from"
            },
            "image_size": {
                "description": "Size preset for the generated image"
            },
            "guidance_scale": {
                "description": "How strictly to follow the prompt. Higher values are more literal"
            },
            "num_inference_steps": {
                "description": "Number of denoising steps. More steps typically improve quality"
            },
            "seed": {
                "description": "Seed for reproducible results. Use -1 for random"
            },
            "num_images": {
                "description": "Number of images to generate"
            },
            "enable_safety_checker": {
                "description": "Enable safety checker to filter unsafe content"
            },
            "safety_tolerance": {
                "description": "Safety checker tolerance level (1-6). Higher is more permissive"
            },
            "output_format": {
                "description": "Output image format (jpeg or png)"
            }
        },
        "basic_fields": ["prompt", "image_size", "guidance_scale"]
    },
    
    "fal-ai/flux-pro/v1.1-ultra": {
        "class_name": "FluxV1ProUltra",
        "docstring": "FLUX.1 Pro Ultra delivers the highest quality image generation with enhanced detail and realism.",
        "tags": ["image", "generation", "flux", "pro", "ultra", "text-to-image", "txt2img"],
        "use_cases": [
            "Generate ultra-high quality photorealistic images",
            "Create professional photography-grade visuals",
            "Produce detailed product renders",
            "Generate premium marketing materials",
            "Create artistic masterpieces with fine details"
        ],
        "field_overrides": {
            "prompt": {
                "description": "The prompt to generate an image from"
            },
            "image_size": {
                "description": "Size preset for the generated image"
            },
            "guidance_scale": {
                "description": "How strictly to follow the prompt"
            },
            "num_inference_steps": {
                "description": "Number of denoising steps"
            },
            "seed": {
                "description": "Seed for reproducible results. Use -1 for random"
            },
            "num_images": {
                "description": "Number of images to generate"
            },
            "raw": {
                "description": "Generate less processed, more natural results"
            },
            "aspect_ratio": {
                "description": "Aspect ratio for the generated image"
            },
            "image_prompt_strength": {
                "description": "Strength of image prompt influence (0-1)"
            }
        },
        "basic_fields": ["prompt", "image_size", "aspect_ratio"]
    },
    
    "fal-ai/flux-lora": {
        "class_name": "FluxLora",
        "docstring": "FLUX with LoRA support enables fine-tuned image generation using custom LoRA models for specific styles or subjects.",
        "tags": ["image", "generation", "flux", "lora", "fine-tuning", "text-to-image", "txt2img"],
        "use_cases": [
            "Generate images with custom artistic styles",
            "Create consistent characters across images",
            "Apply brand-specific visual styles",
            "Generate images with specialized subjects",
            "Combine multiple LoRA models for unique results"
        ],
        "field_overrides": {
            "prompt": {
                "description": "The prompt to generate an image from"
            },
            "image_size": {
                "description": "Size preset for the generated image"
            },
            "num_inference_steps": {
                "description": "Number of denoising steps"
            },
            "guidance_scale": {
                "description": "How strictly to follow the prompt"
            },
            "loras": {
                "description": "List of LoRA models to apply with their weights"
            },
            "seed": {
                "description": "Seed for reproducible results. Use -1 for random"
            },
            "enable_safety_checker": {
                "description": "Enable safety checker to filter unsafe content"
            }
        },
        "basic_fields": ["prompt", "loras", "image_size"]
    },
    
    "fal-ai/ideogram/v2": {
        "class_name": "IdeogramV2",
        "docstring": "Ideogram V2 is a state-of-the-art image generation model optimized for commercial and creative use, featuring exceptional typography handling and realistic outputs.",
        "tags": ["image", "generation", "ai", "typography", "realistic", "text-to-image", "txt2img"],
        "use_cases": [
            "Create commercial artwork and designs",
            "Generate realistic product visualizations",
            "Design marketing materials with text",
            "Produce high-quality illustrations",
            "Create brand assets and logos"
        ],
        "field_overrides": {
            "prompt": {
                "description": "The prompt to generate an image from"
            },
            "aspect_ratio": {
                "description": "The aspect ratio of the generated image"
            },
            "expand_prompt": {
                "description": "Whether to expand the prompt with MagicPrompt functionality"
            },
            "style": {
                "description": "The style of the generated image"
            },
            "negative_prompt": {
                "description": "A negative prompt to avoid in the generated image"
            },
            "seed": {
                "description": "Seed for reproducible results. Use -1 for random"
            }
        },
        "basic_fields": ["prompt", "aspect_ratio", "style"]
    },
    
    "fal-ai/ideogram/v2/turbo": {
        "class_name": "IdeogramV2Turbo",
        "docstring": "Ideogram V2 Turbo offers faster image generation with the same exceptional quality and typography handling as V2.",
        "tags": ["image", "generation", "ai", "typography", "realistic", "fast", "text-to-image", "txt2img"],
        "use_cases": [
            "Rapidly generate commercial designs",
            "Quick iteration on marketing materials",
            "Fast prototyping of visual concepts",
            "Real-time design exploration",
            "Efficient batch generation of branded content"
        ],
        "field_overrides": {
            "prompt": {
                "description": "The prompt to generate an image from"
            },
            "aspect_ratio": {
                "description": "The aspect ratio of the generated image"
            },
            "expand_prompt": {
                "description": "Whether to expand the prompt with MagicPrompt functionality"
            },
            "style": {
                "description": "The style of the generated image"
            },
            "negative_prompt": {
                "description": "A negative prompt to avoid in the generated image"
            },
            "seed": {
                "description": "Seed for reproducible results. Use -1 for random"
            }
        },
        "basic_fields": ["prompt", "aspect_ratio", "style"]
    },
    
    "fal-ai/recraft-v3": {
        "class_name": "RecraftV3",
        "docstring": "Recraft V3 is a powerful image generation model with exceptional control over style and colors, ideal for brand consistency and design work.",
        "tags": ["image", "generation", "design", "branding", "style", "text-to-image", "txt2img"],
        "use_cases": [
            "Create brand-consistent visual assets",
            "Generate designs with specific color palettes",
            "Produce stylized illustrations and artwork",
            "Design marketing materials with brand colors",
            "Create cohesive visual content series"
        ],
        "field_overrides": {
            "prompt": {
                "description": "The prompt to generate an image from"
            },
            "image_size": {
                "description": "Size preset for the generated image"
            },
            "style": {
                "description": "Visual style preset for the generated image"
            },
            "colors": {
                "description": "Specific color palette to use in the generation"
            },
            "style_id": {
                "description": "Custom style ID for brand-specific styles"
            }
        },
        "basic_fields": ["prompt", "style", "colors"]
    },
    
    "fal-ai/stable-diffusion-v35-large": {
        "class_name": "StableDiffusionV35Large",
        "docstring": "Stable Diffusion 3.5 Large is a powerful open-weight model with excellent prompt adherence and diverse output capabilities.",
        "tags": ["image", "generation", "stable-diffusion", "open-source", "text-to-image", "txt2img"],
        "use_cases": [
            "Generate diverse artistic styles",
            "Create high-quality illustrations",
            "Produce photorealistic images",
            "Generate concept art and designs",
            "Create custom visual content"
        ],
        "field_overrides": {
            "prompt": {
                "description": "The prompt to generate an image from"
            },
            "negative_prompt": {
                "description": "Elements to avoid in the generated image"
            },
            "aspect_ratio": {
                "description": "The aspect ratio of the generated image"
            }
        },
        "basic_fields": ["prompt", "negative_prompt", "aspect_ratio"]
    },
    
    "fal-ai/flux-pro/new": {
        "class_name": "FluxProNew",
        "docstring": "FLUX.1 Pro New is the latest version of the professional FLUX model with enhanced capabilities and improved output quality.",
        "tags": ["image", "generation", "flux", "professional", "text-to-image", "txt2img"],
        "use_cases": [
            "Generate professional-grade marketing visuals",
            "Create high-quality product renders",
            "Produce detailed architectural visualizations",
            "Design premium brand assets",
            "Generate photorealistic commercial imagery"
        ],
        "field_overrides": {
            "prompt": {
                "description": "The prompt to generate an image from"
            },
            "image_size": {
                "python_type": "ImageSizePreset",
                "default_value": "ImageSizePreset.LANDSCAPE_4_3",
                "description": "Size preset for the generated image"
            },
            "seed": {
                "python_type": "int",
                "default_value": "-1",
                "description": "Seed for reproducible results. Use -1 for random"
            }
        },
        "basic_fields": ["prompt", "image_size"]
    },
    
    "fal-ai/flux-2/turbo": {
        "class_name": "Flux2Turbo",
        "docstring": "FLUX.2 Turbo is a blazing-fast image generation model optimized for speed without sacrificing quality, ideal for real-time applications.",
        "tags": ["image", "generation", "flux", "fast", "turbo", "text-to-image", "txt2img"],
        "use_cases": [
            "Real-time image generation for interactive apps",
            "Rapid prototyping of visual concepts",
            "Generate multiple variations instantly",
            "Live visual effects and augmented reality",
            "High-throughput batch image processing"
        ],
        "field_overrides": {
            "prompt": {
                "description": "The prompt to generate an image from"
            },
            "image_size": {
                "python_type": "ImageSizePreset",
                "default_value": "ImageSizePreset.LANDSCAPE_4_3",
                "description": "Size preset for the generated image"
            },
            "num_images": {
                "description": "Number of images to generate"
            },
            "seed": {
                "python_type": "int",
                "default_value": "-1",
                "description": "Seed for reproducible results. Use -1 for random"
            }
        },
        "basic_fields": ["prompt", "image_size", "num_images"]
    },
    
    "fal-ai/flux-2/flash": {
        "class_name": "Flux2Flash",
        "docstring": "FLUX.2 Flash is an ultra-fast variant of FLUX.2 designed for instant image generation with minimal latency.",
        "tags": ["image", "generation", "flux", "ultra-fast", "flash", "text-to-image", "txt2img"],
        "use_cases": [
            "Instant preview generation for user interfaces",
            "Real-time collaborative design tools",
            "Lightning-fast concept exploration",
            "High-speed batch processing",
            "Interactive gaming and entertainment applications"
        ],
        "field_overrides": {
            "prompt": {
                "description": "The prompt to generate an image from"
            },
            "image_size": {
                "python_type": "ImageSizePreset",
                "default_value": "ImageSizePreset.LANDSCAPE_4_3",
                "description": "Size preset for the generated image"
            },
            "seed": {
                "python_type": "int",
                "default_value": "-1",
                "description": "Seed for reproducible results. Use -1 for random"
            }
        },
        "basic_fields": ["prompt", "image_size"]
    },
    
    "fal-ai/ideogram/v3": {
        "class_name": "IdeogramV3",
        "docstring": "Ideogram V3 is the latest generation with enhanced text rendering, superior image quality, and expanded creative controls.",
        "tags": ["image", "generation", "ideogram", "typography", "text-rendering", "text-to-image", "txt2img"],
        "use_cases": [
            "Create professional graphics with embedded text",
            "Design social media posts with perfect typography",
            "Generate logos and brand identities",
            "Produce marketing materials with text overlays",
            "Create educational content with clear text"
        ],
        "field_overrides": {
            "prompt": {
                "description": "The prompt to generate an image from"
            },
            "aspect_ratio": {
                "description": "The aspect ratio of the generated image"
            },
            "style": {
                "description": "The style preset for the generated image"
            },
            "expand_prompt": {
                "description": "Automatically enhance the prompt for better results"
            }
        },
        "basic_fields": ["prompt", "aspect_ratio", "style"]
    },
    
    "fal-ai/omnigen-v1": {
        "class_name": "OmniGenV1",
        "docstring": "OmniGen V1 is a versatile unified model for multi-modal image generation and editing with text, supporting complex compositional tasks.",
        "tags": ["image", "generation", "multi-modal", "editing", "unified", "text-to-image", "txt2img"],
        "use_cases": [
            "Generate images with multiple input modalities",
            "Edit existing images with text instructions",
            "Create complex compositional scenes",
            "Combine text and image inputs for generation",
            "Perform advanced image manipulations"
        ],
        "field_overrides": {
            "prompt": {
                "description": "The prompt to generate or edit an image"
            },
            "guidance_scale": {
                "description": "How strictly to follow the prompt and inputs"
            },
            "num_inference_steps": {
                "description": "Number of denoising steps for generation quality"
            },
            "seed": {
                "python_type": "int",
                "default_value": "-1",
                "description": "Seed for reproducible results. Use -1 for random"
            }
        },
        "basic_fields": ["prompt", "guidance_scale", "num_inference_steps"]
    },
    
    "fal-ai/sana": {
        "class_name": "Sana",
        "docstring": "Sana is an efficient high-resolution image generation model that balances quality and speed for practical applications.",
        "tags": ["image", "generation", "efficient", "high-resolution", "text-to-image", "txt2img"],
        "use_cases": [
            "Generate high-resolution images efficiently",
            "Create detailed artwork with good performance",
            "Produce quality visuals with limited compute",
            "Generate images for web and mobile applications",
            "Balanced quality-speed image production"
        ],
        "field_overrides": {
            "prompt": {
                "description": "The prompt to generate an image from"
            },
            "negative_prompt": {
                "description": "Elements to avoid in the generated image"
            },
            "image_size": {
                "python_type": "ImageSizePreset",
                "default_value": "ImageSizePreset.LANDSCAPE_4_3",
                "description": "Size preset for the generated image"
            },
            "guidance_scale": {
                "description": "How strictly to follow the prompt"
            },
            "num_inference_steps": {
                "description": "Number of denoising steps"
            }
        },
        "basic_fields": ["prompt", "image_size", "guidance_scale"]
    }
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
