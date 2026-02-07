"""
Configuration for image_to_image module.

This config file defines overrides and customizations for image-to-image nodes.
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
        "enum_overrides": {
            "Style": "RecraftV3ImageToImageStyle"
        },
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

    # Hunyuan Image Family
    "fal-ai/hunyuan-image/v3/instruct/edit": {
        "class_name": "HunyuanImageV3InstructEdit",
        "docstring": "Hunyuan Image v3 Instruct Edit allows precise image editing through natural language instructions with advanced understanding.",
        "tags": ["image", "editing", "hunyuan", "instruct", "ai-editing"],
        "use_cases": [
            "Edit images using natural language instructions",
            "Modify specific elements in photos with text commands",
            "Apply precise adjustments through conversational editing",
            "Transform images with instruction-based control",
            "Create variations with detailed text guidance"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # Qwen Image Family - Max Edit
    "fal-ai/qwen-image-max/edit": {
        "class_name": "QwenImageMaxEdit",
        "docstring": "Qwen Image Max Edit provides powerful image editing capabilities with advanced AI understanding and high-quality results.",
        "tags": ["image", "editing", "qwen", "max", "ai-editing"],
        "use_cases": [
            "Edit images with advanced AI understanding",
            "Apply complex modifications to photos",
            "Transform images with high-quality results",
            "Create professional edits with natural prompts",
            "Modify images with precise control"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # Qwen Image Family - 2511 Series
    "fal-ai/qwen-image-edit-2511": {
        "class_name": "QwenImageEdit2511",
        "docstring": "Qwen Image Edit 2511 provides state-of-the-art image editing with latest AI advancements and improved quality.",
        "tags": ["image", "editing", "qwen", "2511", "latest"],
        "use_cases": [
            "Edit images with latest Qwen technology",
            "Apply advanced modifications to photos",
            "Create high-quality edits with AI assistance",
            "Transform images with cutting-edge models",
            "Produce professional image modifications"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "fal-ai/qwen-image-edit-2511/lora": {
        "class_name": "QwenImageEdit2511Lora",
        "docstring": "Qwen Image Edit 2511 with LoRA support enables custom-trained models for specialized editing tasks.",
        "tags": ["image", "editing", "qwen", "lora", "custom"],
        "use_cases": [
            "Edit images with custom-trained models",
            "Apply specialized modifications using LoRA",
            "Create domain-specific edits",
            "Transform images with fine-tuned models",
            "Produce customized image modifications"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "fal-ai/qwen-image-edit-2511-multiple-angles": {
        "class_name": "QwenImageEdit2511MultipleAngles",
        "docstring": "Qwen Image Edit 2511 Multiple Angles generates images from different viewpoints based on a single input image.",
        "tags": ["image", "editing", "qwen", "multi-angle", "viewpoint"],
        "use_cases": [
            "Generate multiple viewpoints from single image",
            "Create product views from different angles",
            "Visualize objects from various perspectives",
            "Produce multi-angle image sets",
            "Transform images to show different sides"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # Qwen Image Family - 2509 Series
    "fal-ai/qwen-image-edit-2509": {
        "class_name": "QwenImageEdit2509",
        "docstring": "Qwen Image Edit 2509 provides powerful image editing with advanced AI capabilities and high-quality output.",
        "tags": ["image", "editing", "qwen", "2509", "ai-editing"],
        "use_cases": [
            "Edit images with Qwen 2509 technology",
            "Apply sophisticated modifications to photos",
            "Create quality edits with AI assistance",
            "Transform images with advanced models",
            "Produce professional image changes"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "fal-ai/qwen-image-edit-2509-lora": {
        "class_name": "QwenImageEdit2509Lora",
        "docstring": "Qwen Image Edit 2509 with LoRA enables fine-tuned models for specialized image editing applications.",
        "tags": ["image", "editing", "qwen", "lora", "fine-tuned"],
        "use_cases": [
            "Edit images with fine-tuned models",
            "Apply custom modifications using LoRA",
            "Create specialized edits for specific domains",
            "Transform images with trained models",
            "Produce tailored image modifications"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # Qwen Image Layered
    "fal-ai/qwen-image-layered": {
        "class_name": "QwenImageLayered",
        "docstring": "Qwen Image Layered provides layer-based image editing for complex compositions and precise control.",
        "tags": ["image", "editing", "qwen", "layered", "composition"],
        "use_cases": [
            "Edit images with layer-based control",
            "Create complex compositions",
            "Apply modifications to specific layers",
            "Build multi-layer image edits",
            "Produce sophisticated image compositions"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "fal-ai/qwen-image-layered/lora": {
        "class_name": "QwenImageLayeredLora",
        "docstring": "Qwen Image Layered with LoRA combines layer-based editing with custom-trained models for specialized tasks.",
        "tags": ["image", "editing", "qwen", "layered", "lora"],
        "use_cases": [
            "Edit layered images with custom models",
            "Create specialized layer compositions",
            "Apply fine-tuned modifications",
            "Build complex edits with trained models",
            "Produce custom layer-based results"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # FLUX-2 Klein Family - Base Edit
    "fal-ai/flux-2/klein/4b/base/edit": {
        "class_name": "Flux2Klein4BBaseEdit",
        "docstring": "FLUX-2 Klein 4B Base Edit provides fast image editing with the 4-billion parameter model.",
        "tags": ["image", "editing", "flux-2", "klein", "4b"],
        "use_cases": [
            "Edit images with FLUX-2 Klein 4B",
            "Apply fast modifications to photos",
            "Create quick edits with AI assistance",
            "Transform images efficiently",
            "Produce rapid image modifications"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "fal-ai/flux-2/klein/4b/base/edit/lora": {
        "class_name": "Flux2Klein4BBaseEditLora",
        "docstring": "FLUX-2 Klein 4B Base Edit with LoRA enables custom-trained models for specialized editing.",
        "tags": ["image", "editing", "flux-2", "klein", "4b", "lora"],
        "use_cases": [
            "Edit images with custom FLUX-2 models",
            "Apply specialized modifications using LoRA",
            "Create domain-specific edits",
            "Transform images with fine-tuned 4B model",
            "Produce customized modifications"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "fal-ai/flux-2/klein/9b/base/edit": {
        "class_name": "Flux2Klein9BBaseEdit",
        "docstring": "FLUX-2 Klein 9B Base Edit provides high-quality image editing with the 9-billion parameter model.",
        "tags": ["image", "editing", "flux-2", "klein", "9b"],
        "use_cases": [
            "Edit images with FLUX-2 Klein 9B",
            "Apply high-quality modifications to photos",
            "Create advanced edits with powerful AI",
            "Transform images with superior quality",
            "Produce professional image modifications"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "fal-ai/flux-2/klein/9b/base/edit/lora": {
        "class_name": "Flux2Klein9BBaseEditLora",
        "docstring": "FLUX-2 Klein 9B Base Edit with LoRA combines powerful editing with custom-trained models.",
        "tags": ["image", "editing", "flux-2", "klein", "9b", "lora"],
        "use_cases": [
            "Edit images with custom 9B models",
            "Apply specialized high-quality modifications",
            "Create professional custom edits",
            "Transform images with fine-tuned powerful model",
            "Produce advanced customized results"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # FLUX-2 Klein Family - Standard Edit
    "fal-ai/flux-2/klein/4b/edit": {
        "class_name": "Flux2Klein4BEdit",
        "docstring": "FLUX-2 Klein 4B Edit provides efficient image editing with the streamlined 4-billion parameter model.",
        "tags": ["image", "editing", "flux-2", "klein", "4b", "efficient"],
        "use_cases": [
            "Edit images efficiently with FLUX-2",
            "Apply quick modifications to photos",
            "Create fast edits for rapid workflows",
            "Transform images with streamlined model",
            "Produce quick image modifications"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "fal-ai/flux-2/klein/9b/edit": {
        "class_name": "Flux2Klein9BEdit",
        "docstring": "FLUX-2 Klein 9B Edit provides advanced image editing with the full 9-billion parameter model.",
        "tags": ["image", "editing", "flux-2", "klein", "9b", "advanced"],
        "use_cases": [
            "Edit images with advanced FLUX-2 model",
            "Apply sophisticated modifications",
            "Create high-quality edits",
            "Transform images with powerful AI",
            "Produce superior image modifications"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # FLUX-2 Other Variants
    "fal-ai/flux-2/flash/edit": {
        "class_name": "Flux2FlashEdit",
        "docstring": "FLUX-2 Flash Edit provides ultra-fast image editing for rapid iteration and quick modifications.",
        "tags": ["image", "editing", "flux-2", "flash", "ultra-fast"],
        "use_cases": [
            "Edit images with ultra-fast processing",
            "Apply instant modifications to photos",
            "Create rapid edits for quick turnaround",
            "Transform images at maximum speed",
            "Produce instant image modifications"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "fal-ai/flux-2/turbo/edit": {
        "class_name": "Flux2TurboEdit",
        "docstring": "FLUX-2 Turbo Edit provides accelerated image editing with balanced quality and speed.",
        "tags": ["image", "editing", "flux-2", "turbo", "fast"],
        "use_cases": [
            "Edit images with turbo speed",
            "Apply fast modifications with good quality",
            "Create quick edits efficiently",
            "Transform images rapidly",
            "Produce fast quality modifications"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "fal-ai/flux-2-max/edit": {
        "class_name": "Flux2MaxEdit",
        "docstring": "FLUX-2 Max Edit provides maximum quality image editing with the most advanced FLUX-2 model.",
        "tags": ["image", "editing", "flux-2", "max", "premium"],
        "use_cases": [
            "Edit images with maximum quality",
            "Apply premium modifications to photos",
            "Create professional-grade edits",
            "Transform images with best quality",
            "Produce highest quality modifications"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "fal-ai/flux-2-flex/edit": {
        "class_name": "Flux2FlexEdit",
        "docstring": "FLUX-2 Flex Edit provides flexible image editing with customizable parameters and versatile control.",
        "tags": ["image", "editing", "flux-2", "flex", "versatile"],
        "use_cases": [
            "Edit images with flexible controls",
            "Apply customizable modifications",
            "Create versatile edits",
            "Transform images with adaptable settings",
            "Produce flexible image modifications"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # FLUX-2 LoRA Gallery
    "fal-ai/flux-2-lora-gallery/virtual-tryon": {
        "class_name": "Flux2LoraGalleryVirtualTryon",
        "docstring": "FLUX-2 LoRA Gallery Virtual Try-on enables realistic clothing and accessory visualization on people.",
        "tags": ["image", "editing", "flux-2", "virtual-tryon", "fashion"],
        "use_cases": [
            "Visualize clothing on models",
            "Try on accessories virtually",
            "Create fashion previews",
            "Test product appearances",
            "Generate try-on images"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "fal-ai/flux-2-lora-gallery/multiple-angles": {
        "class_name": "Flux2LoraGalleryMultipleAngles",
        "docstring": "FLUX-2 LoRA Gallery Multiple Angles generates images from different viewpoints for comprehensive visualization.",
        "tags": ["image", "editing", "flux-2", "multi-angle", "viewpoint"],
        "use_cases": [
            "Generate multiple product angles",
            "Create viewpoint variations",
            "Visualize objects from different sides",
            "Produce multi-angle image sets",
            "Generate comprehensive views"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "fal-ai/flux-2-lora-gallery/face-to-full-portrait": {
        "class_name": "Flux2LoraGalleryFaceToFullPortrait",
        "docstring": "FLUX-2 LoRA Gallery Face to Full Portrait expands face crops into complete portrait images.",
        "tags": ["image", "editing", "flux-2", "portrait", "expansion"],
        "use_cases": [
            "Expand face crops to full portraits",
            "Generate complete portrait from face",
            "Create full-body images from headshots",
            "Extend facial images to portraits",
            "Produce complete portrait compositions"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "fal-ai/flux-2-lora-gallery/add-background": {
        "class_name": "Flux2LoraGalleryAddBackground",
        "docstring": "FLUX-2 LoRA Gallery Add Background places subjects in new environments with realistic integration.",
        "tags": ["image", "editing", "flux-2", "background", "compositing"],
        "use_cases": [
            "Add backgrounds to cutout images",
            "Place subjects in new environments",
            "Create realistic background compositions",
            "Generate contextual settings",
            "Produce integrated background images"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # Bria FIBO Edit Suite
    "bria/fibo-edit/edit": {
        "class_name": "BriaFiboEdit",
        "docstring": "Bria FIBO Edit provides general-purpose image editing with AI-powered modifications and enhancements.",
        "tags": ["image", "editing", "bria", "fibo", "general"],
        "use_cases": [
            "Edit images with general-purpose AI",
            "Apply various modifications to photos",
            "Create edited versions of images",
            "Transform images with flexible edits",
            "Produce AI-powered modifications"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "bria/fibo-edit/add_object_by_text": {
        "class_name": "BriaFiboEditAddObjectByText",
        "docstring": "Bria FIBO Edit Add Object by Text inserts new objects into images using text descriptions.",
        "tags": ["image", "editing", "bria", "fibo", "object-insertion"],
        "use_cases": [
            "Add objects to images with text",
            "Insert elements using descriptions",
            "Place new items in scenes",
            "Augment images with additional objects",
            "Generate object additions"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "bria/fibo-edit/erase_by_text": {
        "class_name": "BriaFiboEditEraseByText",
        "docstring": "Bria FIBO Edit Erase by Text removes objects from images using natural language descriptions.",
        "tags": ["image", "editing", "bria", "fibo", "object-removal"],
        "use_cases": [
            "Remove objects using text descriptions",
            "Erase unwanted elements from photos",
            "Clean up images by describing what to remove",
            "Delete specific items from scenes",
            "Remove objects with natural language"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "bria/fibo-edit/replace_object_by_text": {
        "class_name": "BriaFiboEditReplaceObjectByText",
        "docstring": "Bria FIBO Edit Replace Object by Text replaces objects in images with new ones specified by text.",
        "tags": ["image", "editing", "bria", "fibo", "object-replacement"],
        "use_cases": [
            "Replace objects using text descriptions",
            "Swap elements in photos",
            "Change specific items in scenes",
            "Transform objects with text guidance",
            "Substitute objects with new ones"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "bria/fibo-edit/blend": {
        "class_name": "BriaFiboEditBlend",
        "docstring": "Bria FIBO Edit Blend seamlessly combines multiple images or elements with natural transitions.",
        "tags": ["image", "editing", "bria", "fibo", "blending"],
        "use_cases": [
            "Blend multiple images together",
            "Create seamless compositions",
            "Merge elements naturally",
            "Combine images with smooth transitions",
            "Generate blended composites"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "bria/fibo-edit/colorize": {
        "class_name": "BriaFiboEditColorize",
        "docstring": "Bria FIBO Edit Colorize adds realistic colors to grayscale or black-and-white images.",
        "tags": ["image", "editing", "bria", "fibo", "colorization"],
        "use_cases": [
            "Colorize black and white photos",
            "Add colors to grayscale images",
            "Restore color in old photographs",
            "Transform monochrome to color",
            "Generate colored versions of grayscale images"
        ],
        "basic_fields": ["image"]
    },

    "bria/fibo-edit/restore": {
        "class_name": "BriaFiboEditRestore",
        "docstring": "Bria FIBO Edit Restore repairs and enhances damaged or degraded images with AI reconstruction.",
        "tags": ["image", "editing", "bria", "fibo", "restoration"],
        "use_cases": [
            "Restore damaged photographs",
            "Repair degraded images",
            "Enhance old photo quality",
            "Fix scratches and artifacts",
            "Reconstruct missing image parts"
        ],
        "basic_fields": ["image"]
    },

    "bria/fibo-edit/restyle": {
        "class_name": "BriaFiboEditRestyle",
        "docstring": "Bria FIBO Edit Restyle transforms images with artistic style transfers and visual aesthetics.",
        "tags": ["image", "editing", "bria", "fibo", "style-transfer"],
        "use_cases": [
            "Apply artistic styles to images",
            "Transform photos with new aesthetics",
            "Create stylized versions of images",
            "Generate artistic variations",
            "Produce style-transferred images"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "bria/fibo-edit/relight": {
        "class_name": "BriaFiboEditRelight",
        "docstring": "Bria FIBO Edit Relight adjusts lighting conditions in images for dramatic or natural effects.",
        "tags": ["image", "editing", "bria", "fibo", "relighting"],
        "use_cases": [
            "Adjust lighting in photos",
            "Change illumination conditions",
            "Create dramatic lighting effects",
            "Relight scenes for better ambiance",
            "Transform lighting in images"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "bria/fibo-edit/reseason": {
        "class_name": "BriaFiboEditReseason",
        "docstring": "Bria FIBO Edit Reseason changes the seasonal appearance of outdoor scenes in images.",
        "tags": ["image", "editing", "bria", "fibo", "seasonal"],
        "use_cases": [
            "Change seasons in outdoor photos",
            "Transform summer to winter scenes",
            "Modify seasonal appearance",
            "Create seasonal variations",
            "Generate different season versions"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "bria/fibo-edit/rewrite_text": {
        "class_name": "BriaFiboEditRewriteText",
        "docstring": "Bria FIBO Edit Rewrite Text modifies or replaces text content within images naturally.",
        "tags": ["image", "editing", "bria", "fibo", "text-editing"],
        "use_cases": [
            "Change text in images",
            "Replace written content in photos",
            "Modify signs and labels",
            "Update text naturally in scenes",
            "Edit textual elements in images"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "bria/fibo-edit/sketch_to_colored_image": {
        "class_name": "BriaFiboEditSketchToColoredImage",
        "docstring": "Bria FIBO Edit Sketch to Colored Image transforms sketches and line art into full-color images.",
        "tags": ["image", "editing", "bria", "fibo", "sketch-to-image"],
        "use_cases": [
            "Convert sketches to colored images",
            "Transform line art to full color",
            "Generate colored versions of drawings",
            "Create realistic images from sketches",
            "Produce colored artwork from outlines"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # GLM Image
    "fal-ai/glm-image/image-to-image": {
        "class_name": "GlmImageImageToImage",
        "docstring": "GLM Image image-to-image transforms and modifies images using advanced AI understanding.",
        "tags": ["image", "transformation", "glm", "ai-editing"],
        "use_cases": [
            "Transform images with GLM AI",
            "Apply modifications using advanced understanding",
            "Create variations with GLM model",
            "Generate modified versions",
            "Produce AI-powered transformations"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # GPT Image
    "fal-ai/gpt-image-1.5/edit": {
        "class_name": "GptImage15Edit",
        "docstring": "GPT Image 1.5 Edit provides intelligent image editing with GPT-powered understanding and control.",
        "tags": ["image", "editing", "gpt", "intelligent", "ai-editing"],
        "use_cases": [
            "Edit images with GPT intelligence",
            "Apply smart modifications to photos",
            "Create intelligent edits",
            "Transform images with language understanding",
            "Produce GPT-powered modifications"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # Z-Image Turbo Family
    "fal-ai/z-image/turbo/image-to-image": {
        "class_name": "ZImageTurboImageToImage",
        "docstring": "Z-Image Turbo image-to-image provides fast image transformations with quality output.",
        "tags": ["image", "transformation", "z-image", "turbo", "fast"],
        "use_cases": [
            "Transform images quickly with Z-Image",
            "Apply fast modifications to photos",
            "Create rapid image variations",
            "Generate speedy transformations",
            "Produce quick image modifications"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "fal-ai/z-image/turbo/image-to-image/lora": {
        "class_name": "ZImageTurboImageToImageLora",
        "docstring": "Z-Image Turbo image-to-image with LoRA enables fast custom-trained model transformations.",
        "tags": ["image", "transformation", "z-image", "turbo", "lora"],
        "use_cases": [
            "Transform images with custom Z-Image models",
            "Apply fast specialized modifications",
            "Create rapid custom edits",
            "Generate quick customized transformations",
            "Produce fast fine-tuned modifications"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "fal-ai/z-image/turbo/inpaint": {
        "class_name": "ZImageTurboInpaint",
        "docstring": "Z-Image Turbo Inpaint fills masked regions in images quickly with contextually appropriate content.",
        "tags": ["image", "inpainting", "z-image", "turbo", "fast"],
        "use_cases": [
            "Fill masked regions in images quickly",
            "Remove unwanted objects fast",
            "Repair image areas with turbo speed",
            "Generate quick inpainting results",
            "Produce rapid contextual fills"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "fal-ai/z-image/turbo/inpaint/lora": {
        "class_name": "ZImageTurboInpaintLora",
        "docstring": "Z-Image Turbo Inpaint with LoRA provides fast custom-trained inpainting for specialized tasks.",
        "tags": ["image", "inpainting", "z-image", "turbo", "lora"],
        "use_cases": [
            "Inpaint with custom fast models",
            "Fill regions using specialized training",
            "Repair images with custom inpainting",
            "Generate quick custom fills",
            "Produce rapid specialized inpainting"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "fal-ai/z-image/turbo/controlnet": {
        "class_name": "ZImageTurboControlnet",
        "docstring": "Z-Image Turbo ControlNet provides fast controlled image generation with structural guidance.",
        "tags": ["image", "controlnet", "z-image", "turbo", "controlled"],
        "use_cases": [
            "Generate images with fast structural control",
            "Apply quick controlled modifications",
            "Create rapid guided generations",
            "Transform images with fast ControlNet",
            "Produce speedy controlled outputs"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "fal-ai/z-image/turbo/controlnet/lora": {
        "class_name": "ZImageTurboControlnetLora",
        "docstring": "Z-Image Turbo ControlNet with LoRA combines fast controlled generation with custom models.",
        "tags": ["image", "controlnet", "z-image", "turbo", "lora"],
        "use_cases": [
            "Generate with fast custom ControlNet",
            "Apply quick specialized controlled generation",
            "Create rapid custom guided outputs",
            "Transform images with fast custom control",
            "Produce speedy fine-tuned controlled results"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # Face Swap
    "half-moon-ai/ai-face-swap/faceswapimage": {
        "class_name": "AiFaceSwapImage",
        "docstring": "AI Face Swap replaces faces in images with source faces while maintaining natural appearance.",
        "tags": ["image", "face-swap", "ai", "face-manipulation"],
        "use_cases": [
            "Swap faces between images",
            "Replace faces in photos",
            "Create face-swapped variations",
            "Generate face replacement results",
            "Produce face-substituted images"
        ],
        "basic_fields": ["image"]
    },

    # AI Home
    "half-moon-ai/ai-home/style": {
        "class_name": "AiHomeStyle",
        "docstring": "AI Home Style transforms interior spaces with different design styles and aesthetics.",
        "tags": ["image", "interior-design", "style-transfer", "home", "decoration"],
        "use_cases": [
            "Transform interior design styles",
            "Apply different home aesthetics",
            "Create styled room variations",
            "Generate interior design options",
            "Produce home styling transformations"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "half-moon-ai/ai-home/edit": {
        "class_name": "AiHomeEdit",
        "docstring": "AI Home Edit modifies interior spaces with renovations, furniture changes, and design adjustments.",
        "tags": ["image", "interior-design", "editing", "home", "renovation"],
        "use_cases": [
            "Edit interior spaces",
            "Modify room furniture and decor",
            "Create renovation visualizations",
            "Generate design modification options",
            "Produce home editing results"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # AI Baby and Aging
    "half-moon-ai/ai-baby-and-aging-generator/single": {
        "class_name": "AiBabyAndAgingGeneratorSingle",
        "docstring": "AI Baby and Aging Generator Single shows age progression or regression for a single person.",
        "tags": ["image", "aging", "age-progression", "face-manipulation"],
        "use_cases": [
            "Show age progression of person",
            "Generate younger or older versions",
            "Create aging visualizations",
            "Produce age transformation results",
            "Visualize person at different ages"
        ],
        "basic_fields": ["image"]
    },

    "half-moon-ai/ai-baby-and-aging-generator/multi": {
        "class_name": "AiBabyAndAgingGeneratorMulti",
        "docstring": "AI Baby and Aging Generator Multi shows age progression or regression for multiple people in one image.",
        "tags": ["image", "aging", "age-progression", "multi-face"],
        "use_cases": [
            "Show age progression for multiple people",
            "Generate family aging visualizations",
            "Create multi-person aging results",
            "Produce group age transformations",
            "Visualize multiple people at different ages"
        ],
        "basic_fields": ["image"]
    },

    # Wan Image
    "wan/v2.6/image-to-image": {
        "class_name": "WanV26ImageToImage",
        "docstring": "Wan v2.6 image-to-image provides high-quality image transformations with advanced AI capabilities.",
        "tags": ["image", "transformation", "wan", "v2.6", "quality"],
        "use_cases": [
            "Transform images with Wan v2.6",
            "Apply quality modifications to photos",
            "Create high-quality variations",
            "Generate advanced transformations",
            "Produce quality image modifications"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # StepX Edit
    "fal-ai/stepx-edit2": {
        "class_name": "StepxEdit2",
        "docstring": "StepX Edit 2 provides multi-step image editing with progressive refinement and control.",
        "tags": ["image", "editing", "stepx", "progressive", "refinement"],
        "use_cases": [
            "Edit images with progressive steps",
            "Apply multi-stage modifications",
            "Create refined edits gradually",
            "Transform images with step control",
            "Produce progressively refined results"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # Longcat Image
    "fal-ai/longcat-image/edit": {
        "class_name": "LongcatImageEdit",
        "docstring": "Longcat Image Edit transforms images with unique AI-powered modifications and creative control.",
        "tags": ["image", "editing", "longcat", "creative"],
        "use_cases": [
            "Edit images with Longcat AI",
            "Apply creative modifications",
            "Create unique image variations",
            "Transform images creatively",
            "Produce artistic modifications"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # ByteDance SeeDream
    "fal-ai/bytedance/seedream/v4.5/edit": {
        "class_name": "BytedanceSeedreamV45Edit",
        "docstring": "ByteDance SeeDream v4.5 Edit provides advanced image editing with cutting-edge AI technology.",
        "tags": ["image", "editing", "bytedance", "seedream", "v4.5"],
        "use_cases": [
            "Edit images with SeeDream v4.5",
            "Apply advanced modifications",
            "Create high-quality edits",
            "Transform images with latest tech",
            "Produce cutting-edge modifications"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # Vidu
    "fal-ai/vidu/q2/reference-to-image": {
        "class_name": "ViduQ2ReferenceToImage",
        "docstring": "Vidu Q2 Reference-to-Image generates images based on reference images with style and content transfer.",
        "tags": ["image", "generation", "vidu", "reference", "style-transfer"],
        "use_cases": [
            "Generate images from references",
            "Transfer style and content",
            "Create reference-based variations",
            "Transform using reference images",
            "Produce style-transferred results"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # Kling Image
    "fal-ai/kling-image/o1": {
        "class_name": "KlingImageO1",
        "docstring": "Kling Image O1 provides advanced image generation and transformation with optimized quality.",
        "tags": ["image", "generation", "kling", "o1", "optimized"],
        "use_cases": [
            "Generate images with Kling O1",
            "Transform images with optimization",
            "Create optimized quality results",
            "Produce advanced image generations",
            "Generate with balanced quality-speed"
        ],
        "enum_overrides": {
            "AspectRatio": "KlingImageO1AspectRatio"
        },
        "basic_fields": ["image", "prompt"]
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
