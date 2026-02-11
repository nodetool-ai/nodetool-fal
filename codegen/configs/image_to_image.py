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
    "fal-ai/kling-image/o3/image-to-image": {
        "class_name": "KlingImageO3ImageToImage",
        "docstring": "Kling Image O3 transforms images with advanced quality controls and refined detail.",
        "tags": ["image", "transformation", "kling", "o3", "image-to-image", "img2img"],
        "use_cases": [
            "Transform images with Kling O3 quality",
            "Create refined image variations",
            "Apply style transfers with enhanced detail",
            "Generate high-fidelity image edits",
            "Produce consistent image transformations"
        ],
        "enum_overrides": {
            "AspectRatio": "KlingImageO3AspectRatio"
        },
        "basic_fields": ["images", "prompt", "resolution"]
    },

    "fal-ai/qwen-image-edit-2509-lora-gallery/shirt-design": {
        "class_name": "QwenImageEdit2509LoraGalleryShirtDesign",
        "docstring": "Qwen Image Edit 2509 Lora Gallery",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/qwen-image-edit-2509-lora-gallery/remove-lighting": {
        "class_name": "QwenImageEdit2509LoraGalleryRemoveLighting",
        "docstring": "Qwen Image Edit 2509 Lora Gallery",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/qwen-image-edit-2509-lora-gallery/remove-element": {
        "class_name": "QwenImageEdit2509LoraGalleryRemoveElement",
        "docstring": "Qwen Image Edit 2509 Lora Gallery",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/qwen-image-edit-2509-lora-gallery/lighting-restoration": {
        "class_name": "QwenImageEdit2509LoraGalleryLightingRestoration",
        "docstring": "Qwen Image Edit 2509 Lora Gallery",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/qwen-image-edit-2509-lora-gallery/integrate-product": {
        "class_name": "QwenImageEdit2509LoraGalleryIntegrateProduct",
        "docstring": "Qwen Image Edit 2509 Lora Gallery",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora", "professional"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/qwen-image-edit-2509-lora-gallery/group-photo": {
        "class_name": "QwenImageEdit2509LoraGalleryGroupPhoto",
        "docstring": "Qwen Image Edit 2509 Lora Gallery",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/qwen-image-edit-2509-lora-gallery/face-to-full-portrait": {
        "class_name": "QwenImageEdit2509LoraGalleryFaceToFullPortrait",
        "docstring": "Qwen Image Edit 2509 Lora Gallery",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/qwen-image-edit-2509-lora-gallery/add-background": {
        "class_name": "QwenImageEdit2509LoraGalleryAddBackground",
        "docstring": "Qwen Image Edit 2509 Lora Gallery",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/qwen-image-edit-2509-lora-gallery/next-scene": {
        "class_name": "QwenImageEdit2509LoraGalleryNextScene",
        "docstring": "Qwen Image Edit 2509 Lora Gallery",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/qwen-image-edit-2509-lora-gallery/multiple-angles": {
        "class_name": "QwenImageEdit2509LoraGalleryMultipleAngles",
        "docstring": "Qwen Image Edit 2509 Lora Gallery",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/qwen-image-edit-plus-lora-gallery/lighting-restoration": {
        "class_name": "QwenImageEditPlusLoraGalleryLightingRestoration",
        "docstring": "Qwen Image Edit Plus Lora Gallery",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/moondream3-preview/segment": {
        "class_name": "Moondream3PreviewSegment",
        "docstring": "Moondream3 Preview [Segment]",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/flux-2-lora-gallery/apartment-staging": {
        "class_name": "Flux2LoraGalleryApartmentStaging",
        "docstring": "Flux 2 Lora Gallery",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "clarityai/crystal-upscaler": {
        "class_name": "ClarityaiCrystalUpscaler",
        "docstring": "Crystal Upscaler",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/chrono-edit-lora": {
        "class_name": "ChronoEditLora",
        "docstring": "Chrono Edit Lora",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/chrono-edit-lora-gallery/paintbrush": {
        "class_name": "ChronoEditLoraGalleryPaintbrush",
        "docstring": "Chrono Edit Lora Gallery",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/chrono-edit-lora-gallery/upscaler": {
        "class_name": "ChronoEditLoraGalleryUpscaler",
        "docstring": "Chrono Edit Lora Gallery",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/sam-3/image-rle": {
        "class_name": "Sam3ImageRle",
        "docstring": "Sam 3",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/sam-3/image": {
        "class_name": "Sam3Image",
        "docstring": "Segment Anything Model 3",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/gemini-3-pro-image-preview/edit": {
        "class_name": "Gemini3ProImagePreviewEdit",
        "docstring": "Gemini 3 Pro Image Preview",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "professional"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/nano-banana-pro/edit": {
        "class_name": "NanoBananaProEdit",
        "docstring": "Nano Banana Pro",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "professional"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/qwen-image-edit-plus-lora-gallery/multiple-angles": {
        "class_name": "QwenImageEditPlusLoraGalleryMultipleAngles",
        "docstring": "Qwen Image Edit Plus Lora Gallery",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/qwen-image-edit-plus-lora-gallery/shirt-design": {
        "class_name": "QwenImageEditPlusLoraGalleryShirtDesign",
        "docstring": "Qwen Image Edit Plus Lora Gallery",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/qwen-image-edit-plus-lora-gallery/remove-lighting": {
        "class_name": "QwenImageEditPlusLoraGalleryRemoveLighting",
        "docstring": "Qwen Image Edit Plus Lora Gallery",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/qwen-image-edit-plus-lora-gallery/remove-element": {
        "class_name": "QwenImageEditPlusLoraGalleryRemoveElement",
        "docstring": "Qwen Image Edit Plus Lora Gallery",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/qwen-image-edit-plus-lora-gallery/next-scene": {
        "class_name": "QwenImageEditPlusLoraGalleryNextScene",
        "docstring": "Qwen Image Edit Plus Lora Gallery",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/qwen-image-edit-plus-lora-gallery/integrate-product": {
        "class_name": "QwenImageEditPlusLoraGalleryIntegrateProduct",
        "docstring": "Qwen Image Edit Plus Lora Gallery",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora", "professional"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/qwen-image-edit-plus-lora-gallery/group-photo": {
        "class_name": "QwenImageEditPlusLoraGalleryGroupPhoto",
        "docstring": "Qwen Image Edit Plus Lora Gallery",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/qwen-image-edit-plus-lora-gallery/face-to-full-portrait": {
        "class_name": "QwenImageEditPlusLoraGalleryFaceToFullPortrait",
        "docstring": "Qwen Image Edit Plus Lora Gallery",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/qwen-image-edit-plus-lora-gallery/add-background": {
        "class_name": "QwenImageEditPlusLoraGalleryAddBackground",
        "docstring": "Qwen Image Edit Plus Lora Gallery",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/reve/fast/remix": {
        "class_name": "ReveFastRemix",
        "docstring": "Reve",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "fast"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/reve/fast/edit": {
        "class_name": "ReveFastEdit",
        "docstring": "Reve",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "fast"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/image-apps-v2/outpaint": {
        "class_name": "ImageAppsV2Outpaint",
        "docstring": "Image Outpaint",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/flux-vision-upscaler": {
        "class_name": "FluxVisionUpscaler",
        "docstring": "Flux Vision Upscaler",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/emu-3.5-image/edit-image": {
        "class_name": "Emu35ImageEditImage",
        "docstring": "Emu 3.5 Image",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/chrono-edit": {
        "class_name": "ChronoEdit",
        "docstring": "Chrono Edit",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/gpt-image-1-mini/edit": {
        "class_name": "GptImage1MiniEdit",
        "docstring": "GPT Image 1 Mini",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/reve/remix": {
        "class_name": "ReveRemix",
        "docstring": "Reve",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/reve/edit": {
        "class_name": "ReveEdit",
        "docstring": "Reve",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/image2pixel": {
        "class_name": "Image2Pixel",
        "docstring": "Image2Pixel",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/dreamomni2/edit": {
        "class_name": "Dreamomni2Edit",
        "docstring": "DreamOmni2",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/qwen-image-edit-plus-lora": {
        "class_name": "QwenImageEditPlusLora",
        "docstring": "Qwen Image Edit Plus Lora",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/lucidflux": {
        "class_name": "Lucidflux",
        "docstring": "Lucidflux",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/qwen-image-edit/image-to-image": {
        "class_name": "QwenImageEditImageToImage",
        "docstring": "Qwen Image Edit",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/wan-25-preview/image-to-image": {
        "class_name": "Wan25PreviewImageToImage",
        "docstring": "Wan 2.5 Image to Image",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/qwen-image-edit-plus": {
        "class_name": "QwenImageEditPlus",
        "docstring": "Qwen Image Edit Plus",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/seedvr/upscale/image": {
        "class_name": "SeedvrUpscaleImage",
        "docstring": "SeedVR2",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/image-apps-v2/product-holding": {
        "class_name": "ImageAppsV2ProductHolding",
        "docstring": "Product Holding",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "professional"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/image-apps-v2/product-photography": {
        "class_name": "ImageAppsV2ProductPhotography",
        "docstring": "Product Photography",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "professional"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/image-apps-v2/virtual-try-on": {
        "class_name": "ImageAppsV2VirtualTryOn",
        "docstring": "Virtual Try-on",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/image-apps-v2/texture-transform": {
        "class_name": "ImageAppsV2TextureTransform",
        "docstring": "Texture Transform",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/image-apps-v2/relighting": {
        "class_name": "ImageAppsV2Relighting",
        "docstring": "Relighting",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/image-apps-v2/style-transfer": {
        "class_name": "ImageAppsV2StyleTransfer",
        "docstring": "Style Transfer",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/image-apps-v2/photo-restoration": {
        "class_name": "ImageAppsV2PhotoRestoration",
        "docstring": "Photo Restoration",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/image-apps-v2/portrait-enhance": {
        "class_name": "ImageAppsV2PortraitEnhance",
        "docstring": "Portrait Enhance",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/image-apps-v2/photography-effects": {
        "class_name": "ImageAppsV2PhotographyEffects",
        "docstring": "Photography Effects",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/image-apps-v2/perspective": {
        "class_name": "ImageAppsV2Perspective",
        "docstring": "Perspective Change",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/image-apps-v2/object-removal": {
        "class_name": "ImageAppsV2ObjectRemoval",
        "docstring": "Object Removal",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/image-apps-v2/headshot-photo": {
        "class_name": "ImageAppsV2HeadshotPhoto",
        "docstring": "Headshot Generator",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/image-apps-v2/hair-change": {
        "class_name": "ImageAppsV2HairChange",
        "docstring": "Hair Change",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/image-apps-v2/expression-change": {
        "class_name": "ImageAppsV2ExpressionChange",
        "docstring": "Expression Change",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/image-apps-v2/city-teleport": {
        "class_name": "ImageAppsV2CityTeleport",
        "docstring": "City Teleport",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/image-apps-v2/age-modify": {
        "class_name": "ImageAppsV2AgeModify",
        "docstring": "Age Modify",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/image-apps-v2/makeup-application": {
        "class_name": "ImageAppsV2MakeupApplication",
        "docstring": "Makeup Changer",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/qwen-image-edit/inpaint": {
        "class_name": "QwenImageEditInpaint",
        "docstring": "Qwen Image Edit",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/flux/srpo/image-to-image": {
        "class_name": "FluxSrpoImageToImage",
        "docstring": "FLUX.1 SRPO [dev]",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/flux-1/srpo/image-to-image": {
        "class_name": "Flux1SrpoImageToImage",
        "docstring": "FLUX.1 SRPO [dev]",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/qwen-image-edit-lora": {
        "class_name": "QwenImageEditLora",
        "docstring": "Qwen Image Edit Lora",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/vidu/reference-to-image": {
        "class_name": "ViduReferenceToImage",
        "docstring": "Vidu",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/bytedance/seedream/v4/edit": {
        "class_name": "BytedanceSeedreamV4Edit",
        "docstring": "Bytedance Seedream v4 Edit",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/wan/v2.2-a14b/image-to-image": {
        "class_name": "WanV22A14BImageToImage",
        "docstring": "Wan",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/uso": {
        "class_name": "Uso",
        "docstring": "Uso",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/gemini-25-flash-image/edit": {
        "class_name": "Gemini25FlashImageEdit",
        "docstring": "Gemini 2.5 Flash Image",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/qwen-image/image-to-image": {
        "class_name": "QwenImageImageToImage",
        "docstring": "Qwen Image",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "bria/reimagine/3.2": {
        "class_name": "BriaReimagine32",
        "docstring": "Reimagine",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/nano-banana/edit": {
        "class_name": "NanoBananaEdit",
        "docstring": "Nano Banana",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/nextstep-1": {
        "class_name": "Nextstep1",
        "docstring": "Nextstep 1",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/qwen-image-edit": {
        "class_name": "QwenImageEdit",
        "docstring": "Qwen Image Edit",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/ideogram/character/edit": {
        "class_name": "IdeogramCharacterEdit",
        "docstring": "Ideogram V3 Character Edit",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/ideogram/character": {
        "class_name": "IdeogramCharacter",
        "docstring": "Ideogram V3 Character",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/ideogram/character/remix": {
        "class_name": "IdeogramCharacterRemix",
        "docstring": "Ideogram V3 Character Remix",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/flux-krea-lora/inpainting": {
        "class_name": "FluxKreaLoraInpainting",
        "docstring": "FLUX.1 Krea [dev] Inpainting with LoRAs",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/flux-krea-lora/image-to-image": {
        "class_name": "FluxKreaLoraImageToImage",
        "docstring": "FLUX.1 Krea [dev] with LoRAs",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/flux/krea/image-to-image": {
        "class_name": "FluxKreaImageToImage",
        "docstring": "FLUX.1 Krea [dev]",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/flux/krea/redux": {
        "class_name": "FluxKreaRedux",
        "docstring": "FLUX.1 Krea [dev] Redux",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/flux-1/krea/image-to-image": {
        "class_name": "Flux1KreaImageToImage",
        "docstring": "FLUX.1 Krea [dev]",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/flux-1/krea/redux": {
        "class_name": "Flux1KreaRedux",
        "docstring": "FLUX.1 Krea [dev] Redux",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/flux-kontext-lora/inpaint": {
        "class_name": "FluxKontextLoraInpaint",
        "docstring": "Flux Kontext Lora",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/hunyuan_world": {
        "class_name": "Hunyuan_World",
        "docstring": "Hunyuan World",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/image-editing/retouch": {
        "class_name": "ImageEditingRetouch",
        "docstring": "Image Editing",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/hidream-e1-1": {
        "class_name": "HidreamE11",
        "docstring": "Hidream E1 1",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/rife": {
        "class_name": "Rife",
        "docstring": "RIFE",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/film": {
        "class_name": "Film",
        "docstring": "FILM",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/calligrapher": {
        "class_name": "Calligrapher",
        "docstring": "Calligrapher",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/bria/reimagine": {
        "class_name": "BriaReimagine",
        "docstring": "Bria",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/image-editing/realism": {
        "class_name": "ImageEditingRealism",
        "docstring": "Image Editing",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/post-processing/vignette": {
        "class_name": "PostProcessingVignette",
        "docstring": "Post Processing",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "professional"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/post-processing/solarize": {
        "class_name": "PostProcessingSolarize",
        "docstring": "Post Processing",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "professional"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/post-processing/sharpen": {
        "class_name": "PostProcessingSharpen",
        "docstring": "Post Processing",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "professional"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/post-processing/parabolize": {
        "class_name": "PostProcessingParabolize",
        "docstring": "Post Processing",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "professional"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/post-processing/grain": {
        "class_name": "PostProcessingGrain",
        "docstring": "Post Processing",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "professional"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/post-processing/dodge-burn": {
        "class_name": "PostProcessingDodgeBurn",
        "docstring": "Post Processing",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "professional"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/post-processing/dissolve": {
        "class_name": "PostProcessingDissolve",
        "docstring": "Post Processing",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "professional"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/post-processing/desaturate": {
        "class_name": "PostProcessingDesaturate",
        "docstring": "Post Processing",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "professional"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/post-processing/color-tint": {
        "class_name": "PostProcessingColorTint",
        "docstring": "Post Processing",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "professional"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization"

        ],
    },
    "fal-ai/post-processing/color-correction": {
        "class_name": "PostProcessingColorCorrection",
        "docstring": "Adjust color temperature, brightness, contrast, saturation, and gamma values for color correction.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/post-processing/chromatic-aberration": {
        "class_name": "PostProcessingChromaticAberration",
        "docstring": "Create chromatic aberration by shifting red, green, and blue channels horizontally or vertically with customizable shift amounts.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/post-processing/blur": {
        "class_name": "PostProcessingBlur",
        "docstring": "Apply Gaussian or Kuwahara blur effects with adjustable radius and sigma parameters",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-editing/youtube-thumbnails": {
        "class_name": "ImageEditingYoutubeThumbnails",
        "docstring": "Generate YouTube thumbnails with custom text",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/topaz/upscale/image": {
        "class_name": "TopazUpscaleImage",
        "docstring": "Use the powerful and accurate topaz image enhancer to enhance your images.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-editing/broccoli-haircut": {
        "class_name": "ImageEditingBroccoliHaircut",
        "docstring": "Transform your character's hair into broccoli style while keeping the original characters likeness",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-editing/wojak-style": {
        "class_name": "ImageEditingWojakStyle",
        "docstring": "Transform your photos into wojak style while keeping the original characters likeness",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-editing/plushie-style": {
        "class_name": "ImageEditingPlushieStyle",
        "docstring": "Transform your photos into cool plushies while keeping the original characters likeness",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/flux-kontext-lora": {
        "class_name": "FluxKontextLora",
        "docstring": "Fast endpoint for the FLUX.1 Kontext [dev] model with LoRA support, enabling rapid and high-quality image editing using pre-trained LoRA adaptations for specific styles, brand identities, and product-specific outputs.",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/fashn/tryon/v1.6": {
        "class_name": "FashnTryonV16",
        "docstring": "FASHN v1.6 delivers precise virtual try-on capabilities, accurately rendering garment details like text and patterns at 864x1296 resolution from both on-model and flat-lay photo references.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/chain-of-zoom": {
        "class_name": "ChainOfZoom",
        "docstring": "Extreme Super-Resolution via Scale Autoregression and Preference Alignment",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/pasd": {
        "class_name": "Pasd",
        "docstring": "Pixel-Aware Diffusion Model for Realistic Image Super-Resolution and Personalized Stylization",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/object-removal/bbox": {
        "class_name": "ObjectRemovalBbox",
        "docstring": "Removes box-selected objects and their visual effects, seamlessly reconstructing the scene with contextually appropriate content.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/object-removal/mask": {
        "class_name": "ObjectRemovalMask",
        "docstring": "Removes mask-selected objects and their visual effects, seamlessly reconstructing the scene with contextually appropriate content.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/object-removal": {
        "class_name": "ObjectRemoval",
        "docstring": "Removes objects and their visual effects using natural language, replacing them with contextually appropriate content",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/recraft/vectorize": {
        "class_name": "RecraftVectorize",
        "docstring": "Converts a given raster image to SVG format using Recraft model.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/ffmpeg-api/extract-frame": {
        "class_name": "FfmpegApiExtractFrame",
        "docstring": "ffmpeg endpoint for first, middle and last frame extraction from videos",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/luma-photon/flash/modify": {
        "class_name": "LumaPhotonFlashModify",
        "docstring": "Edit images from your prompts using Luma Photon. Photon is the most creative, personalizable, and intelligent visual models for creatives, bringing a step-function change in the cost of high-quality image generation.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/luma-photon/modify": {
        "class_name": "LumaPhotonModify",
        "docstring": "Edit images from your prompts using Luma Photon. Photon is the most creative, personalizable, and intelligent visual models for creatives, bringing a step-function change in the cost of high-quality image generation.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-editing/reframe": {
        "class_name": "ImageEditingReframe",
        "docstring": "The reframe endpoint intelligently adjusts an image's aspect ratio while preserving the main subject's position, composition, pose, and perspective",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-editing/baby-version": {
        "class_name": "ImageEditingBabyVersion",
        "docstring": "Transform any person into their baby version, while preserving the original pose and expression with childlike features.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/luma-photon/flash/reframe": {
        "class_name": "LumaPhotonFlashReframe",
        "docstring": "This advanced tool intelligently expands your visuals, seamlessly blending new content to enhance creativity and adaptability, offering unmatched speed and quality for creators at a fraction of the cost.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/luma-photon/reframe": {
        "class_name": "LumaPhotonReframe",
        "docstring": "Extend and reframe images with Luma Photon Reframe. This advanced tool intelligently expands your visuals, seamlessly blending new content to enhance creativity and adaptability, offering unmatched personalization and quality for creators at a fraction of the cost.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/flux-1/schnell/redux": {
        "class_name": "Flux1SchnellRedux",
        "docstring": "FLUX.1 [schnell] Redux is a high-performance endpoint for the FLUX.1 [schnell] model that enables rapid transformation of existing images, delivering high-quality style transfers and image modifications with the core FLUX capabilities. ",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/flux-1/dev/redux": {
        "class_name": "Flux1DevRedux",
        "docstring": "FLUX.1 [dev] Redux is a high-performance endpoint for the FLUX.1 [dev] model that enables rapid transformation of existing images, delivering high-quality style transfers and image modifications with the core FLUX capabilities.",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/flux-1/dev/image-to-image": {
        "class_name": "Flux1DevImageToImage",
        "docstring": "FLUX.1 [dev] is a 12 billion parameter flow transformer that generates high-quality images from text. It is suitable for personal and commercial use. ",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-editing/text-removal": {
        "class_name": "ImageEditingTextRemoval",
        "docstring": "Remove all text and writing from images while preserving the background and natural appearance.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-editing/photo-restoration": {
        "class_name": "ImageEditingPhotoRestoration",
        "docstring": "Restore and enhance old or damaged photos by removing imperfections, adding color while preserving the original character and details of the image.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-editing/weather-effect": {
        "class_name": "ImageEditingWeatherEffect",
        "docstring": "Add realistic weather effects like snowfall, rain, or fog to your photos while maintaining the scene's mood.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-editing/time-of-day": {
        "class_name": "ImageEditingTimeOfDay",
        "docstring": "Transform your photos to any time of day, from golden hour to midnight, with appropriate lighting and atmosphere.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-editing/style-transfer": {
        "class_name": "ImageEditingStyleTransfer",
        "docstring": "Transform your photos into artistic masterpieces inspired by famous styles like Van Gogh's Starry Night or any artistic style you choose.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-editing/scene-composition": {
        "class_name": "ImageEditingSceneComposition",
        "docstring": "Place your subject in any scene you imagine, from enchanted forests to urban settings, with professional composition and lighting",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-editing/professional-photo": {
        "class_name": "ImageEditingProfessionalPhoto",
        "docstring": "Turn your casual photos into stunning professional studio portraits with perfect lighting and high-end photography style.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-editing/object-removal": {
        "class_name": "ImageEditingObjectRemoval",
        "docstring": "Remove unwanted objects or people from your photos while seamlessly blending the background.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-editing/hair-change": {
        "class_name": "ImageEditingHairChange",
        "docstring": "Experiment with different hairstyles, from bald to any style you can imagine, while maintaining natural lighting and realistic results.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-editing/face-enhancement": {
        "class_name": "ImageEditingFaceEnhancement",
        "docstring": "Enhance facial features with professional retouching while maintaining a natural, realistic look",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-editing/expression-change": {
        "class_name": "ImageEditingExpressionChange",
        "docstring": "Change facial expressions in photos to any emotion you desire, from smiles to serious looks.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-editing/color-correction": {
        "class_name": "ImageEditingColorCorrection",
        "docstring": "Perfect your photos with professional color grading, balanced tones, and vibrant yet natural colors",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-editing/cartoonify": {
        "class_name": "ImageEditingCartoonify",
        "docstring": "Transform your photos into vibrant cool cartoons with bold outlines and rich colors.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-editing/background-change": {
        "class_name": "ImageEditingBackgroundChange",
        "docstring": "Replace your photo's background with any scene you desire, from beach sunsets to urban landscapes, with perfect lighting and shadows",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-editing/age-progression": {
        "class_name": "ImageEditingAgeProgression",
        "docstring": "See how you or others might look at different ages, from younger to older, while preserving core facial features.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/flux-pro/kontext/max/multi": {
        "class_name": "FluxProKontextMaxMulti",
        "docstring": "Experimental version of FLUX.1 Kontext [max] with multi image handling capabilities",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/flux-pro/kontext/multi": {
        "class_name": "FluxProKontextMulti",
        "docstring": "Experimental version of FLUX.1 Kontext [pro] with multi image handling capabilities",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/flux-pro/kontext/max": {
        "class_name": "FluxProKontextMax",
        "docstring": "FLUX.1 Kontext [max] is a model with greatly improved prompt adherence and typography generation meet premium consistency for editing without compromise on speed.   ",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/flux-kontext/dev": {
        "class_name": "FluxKontextDev",
        "docstring": "Frontier image editing model.",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/bagel/edit": {
        "class_name": "BagelEdit",
        "docstring": "Bagel is a 7B parameter multimodal model from Bytedance-Seed that can generate both images and text.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "smoretalk-ai/rembg-enhance": {
        "class_name": "SmoretalkAiRembgEnhance",
        "docstring": "Rembg-enhance is optimized for 2D vector images, 3D graphics, and photos by leveraging matting technology.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/recraft/upscale/creative": {
        "class_name": "RecraftUpscaleCreative",
        "docstring": "Enhances a given raster image using the 'creative upscale' tool, increasing image resolution, making the image sharper and cleaner.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/recraft/upscale/crisp": {
        "class_name": "RecraftUpscaleCrisp",
        "docstring": "Enhances a given raster image using 'crisp upscale' tool, boosting resolution with a focus on refining small details and faces.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/minimax/image-01/subject-reference": {
        "class_name": "MinimaxImage01SubjectReference",
        "docstring": "Generate images from text and a reference image using MiniMax Image-01 for consistent character appearance.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/hidream-i1-full/image-to-image": {
        "class_name": "HidreamI1FullImageToImage",
        "docstring": "HiDream-I1 full is a new open-source image generative foundation model with 17B parameters that achieves state-of-the-art image generation quality within seconds.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/ideogram/v3/reframe": {
        "class_name": "IdeogramV3Reframe",
        "docstring": "Extend existing images with Ideogram V3's reframe feature. Create expanded versions and adaptations while preserving main image and adding new creative directions through prompt guidance.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/ideogram/v3/replace-background": {
        "class_name": "IdeogramV3ReplaceBackground",
        "docstring": "Replace backgrounds existing images with Ideogram V3's replace background feature. Create variations and adaptations while preserving core elements and adding new creative directions through prompt guidance.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/ideogram/v3/remix": {
        "class_name": "IdeogramV3Remix",
        "docstring": "Reimagine existing images with Ideogram V3's remix feature. Create variations and adaptations while preserving core elements and adding new creative directions through prompt guidance.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/step1x-edit": {
        "class_name": "Step1xEdit",
        "docstring": "Step1X-Edit transforms your photos with simple instructions into stunning, professional-quality edits—rivaling top proprietary tools.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image2svg": {
        "class_name": "Image2svg",
        "docstring": "Image2SVG transforms raster images into clean vector graphics, preserving visual quality while enabling scalable, customizable SVG outputs with precise control over detail levels.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/uno": {
        "class_name": "Uno",
        "docstring": "An AI model that transforms input images into new ones based on text prompts, blending reference visuals with your creative directions.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/gpt-image-1/edit-image": {
        "class_name": "GptImage1EditImage",
        "docstring": "OpenAI's latest image generation and editing model: gpt-1-image.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "rundiffusion-fal/juggernaut-flux-lora/inpainting": {
        "class_name": "RundiffusionFalJuggernautFluxLoraInpainting",
        "docstring": "Juggernaut Base Flux LoRA Inpainting by RunDiffusion is a drop-in replacement for Flux [Dev] inpainting that delivers sharper details, richer colors, and enhanced realism to all your LoRAs and LyCORIS with full compatibility.",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/fashn/tryon/v1.5": {
        "class_name": "FashnTryonV15",
        "docstring": "FASHN v1.5 delivers precise virtual try-on capabilities, accurately rendering garment details like text and patterns at 576x864 resolution from both on-model and flat-lay photo references.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/plushify": {
        "class_name": "Plushify",
        "docstring": "Turn any image into a cute plushie!",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/instant-character": {
        "class_name": "InstantCharacter",
        "docstring": "InstantCharacter creates high-quality, consistent characters from text prompts, supporting diverse poses, styles, and appearances with strong identity control.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/cartoonify": {
        "class_name": "Cartoonify",
        "docstring": "Transform images into 3D cartoon artwork using an AI model that applies cartoon stylization while preserving the original image's composition and details.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/finegrain-eraser/mask": {
        "class_name": "FinegrainEraserMask",
        "docstring": "Finegrain Eraser removes any object selected with a mask—along with its shadows, reflections, and lighting artifacts—seamlessly reconstructing the scene with contextually accurate content.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/finegrain-eraser/bbox": {
        "class_name": "FinegrainEraserBbox",
        "docstring": "Finegrain Eraser removes any object selected with a bounding box—along with its shadows, reflections, and lighting artifacts—seamlessly reconstructing the scene with contextually accurate content.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/finegrain-eraser": {
        "class_name": "FinegrainEraser",
        "docstring": "Finegrain Eraser removes objects—along with their shadows, reflections, and lighting artifacts—using only natural language, seamlessly filling the scene with contextually accurate content.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/star-vector": {
        "class_name": "StarVector",
        "docstring": "AI vectorization model that transforms raster images into scalable SVG graphics, preserving visual details while enabling infinite scaling and easy editing capabilities.  ",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/ghiblify": {
        "class_name": "Ghiblify",
        "docstring": "Reimagine and transform your ordinary photos into enchanting Studio Ghibli style artwork",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/thera": {
        "class_name": "Thera",
        "docstring": "Fix low resolution images with fast speed and quality of thera.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/mix-dehaze-net": {
        "class_name": "MixDehazeNet",
        "docstring": "An advanced dehaze model to remove atmospheric haze, restoring clarity and detail in images through intelligent neural network processing.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/gemini-flash-edit": {
        "class_name": "GeminiFlashEdit",
        "docstring": "Gemini Flash Edit is a model that can edit single image using a text prompt and a reference image.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/gemini-flash-edit/multi": {
        "class_name": "GeminiFlashEditMulti",
        "docstring": "Gemini Flash Edit Multi Image is a model that can edit multiple images using a text prompt and a reference image.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/invisible-watermark": {
        "class_name": "InvisibleWatermark",
        "docstring": "Invisible Watermark is a model that can add an invisible watermark to an image.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "rundiffusion-fal/juggernaut-flux/base/image-to-image": {
        "class_name": "RundiffusionFalJuggernautFluxBaseImageToImage",
        "docstring": "Juggernaut Base Flux by RunDiffusion is a drop-in replacement for Flux [Dev] that delivers sharper details, richer colors, and enhanced realism, while instantly boosting LoRAs and LyCORIS with full compatibility.",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "rundiffusion-fal/juggernaut-flux/pro/image-to-image": {
        "class_name": "RundiffusionFalJuggernautFluxProImageToImage",
        "docstring": "Juggernaut Pro Flux by RunDiffusion is the flagship Juggernaut model rivaling some of the most advanced image models available, often surpassing them in realism. It combines Juggernaut Base with RunDiffusion Photo and features enhancements like reduced background blurriness.",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/docres/dewarp": {
        "class_name": "DocresDewarp",
        "docstring": "Enhance wraped, folded documents with the superior quality of docres for sharper, clearer results.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/docres": {
        "class_name": "Docres",
        "docstring": "Enhance low-resolution, blur, shadowed documents with the superior quality of docres for sharper, clearer results.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/swin2sr": {
        "class_name": "Swin2sr",
        "docstring": "Enhance low-resolution images with the superior quality of Swin2SR for sharper, clearer results.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/ideogram/v2a/remix": {
        "class_name": "IdeogramV2aRemix",
        "docstring": "Create variations of existing images with Ideogram V2A Remix while maintaining creative control through prompt guidance.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/ideogram/v2a/turbo/remix": {
        "class_name": "IdeogramV2aTurboRemix",
        "docstring": "Rapidly create image variations with Ideogram V2A Turbo Remix. Fast and efficient reimagining of existing images while maintaining creative control through prompt guidance.",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "fast"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/evf-sam": {
        "class_name": "EvfSam",
        "docstring": "EVF-SAM2 combines natural language understanding with advanced segmentation capabilities, allowing you to precisely mask image regions using intuitive positive and negative text prompts.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/ddcolor": {
        "class_name": "Ddcolor",
        "docstring": "Bring colors into old or new black and white photos with DDColor.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/sam2/auto-segment": {
        "class_name": "Sam2AutoSegment",
        "docstring": "SAM 2 is a model for segmenting images automatically. It can return individual masks or a single mask for the entire image.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/drct-super-resolution": {
        "class_name": "DrctSuperResolution",
        "docstring": "Upscale your images with DRCT-Super-Resolution.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/nafnet/deblur": {
        "class_name": "NafnetDeblur",
        "docstring": "Use NAFNet to fix issues like blurriness and noise in your images. This model specializes in image restoration and can help enhance the overall quality of your photography.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/nafnet/denoise": {
        "class_name": "NafnetDenoise",
        "docstring": "Use NAFNet to fix issues like blurriness and noise in your images. This model specializes in image restoration and can help enhance the overall quality of your photography.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/post-processing": {
        "class_name": "PostProcessing",
        "docstring": "Post Processing is an endpoint that can enhance images using a variety of techniques including grain, blur, sharpen, and more.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/flowedit": {
        "class_name": "Flowedit",
        "docstring": "The model provides you high quality image editing capabilities.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/ben/v2/image": {
        "class_name": "BenV2Image",
        "docstring": "A fast and high quality model for image background removal.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/flux-control-lora-canny/image-to-image": {
        "class_name": "FluxControlLoraCannyImageToImage",
        "docstring": "FLUX Control LoRA Canny is a high-performance endpoint that uses a control image using a Canny edge map to transfer structure to the generated image and another initial image to guide color.",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/flux-control-lora-depth/image-to-image": {
        "class_name": "FluxControlLoraDepthImageToImage",
        "docstring": "FLUX Control LoRA Depth is a high-performance endpoint that uses a control image using a depth map to transfer structure to the generated image and another initial image to guide color.",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/ideogram/upscale": {
        "class_name": "IdeogramUpscale",
        "docstring": "Ideogram Upscale enhances the resolution of the reference image by up to 2X and might enhance the reference image too. Optionally refine outputs with a prompt for guided improvements.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/kling/v1-5/kolors-virtual-try-on": {
        "class_name": "KlingV15KolorsVirtualTryOn",
        "docstring": "Kling Kolors Virtual TryOn v1.5 is a high quality image based Try-On endpoint which can be used for commercial try on.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/flux-lora-canny": {
        "class_name": "FluxLoraCanny",
        "docstring": "Utilize Flux.1 [dev] Controlnet to generate high-quality images with precise control over composition, style, and structure through advanced edge detection and guidance mechanisms.",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/flux-pro/v1/fill-finetuned": {
        "class_name": "FluxProV1FillFinetuned",
        "docstring": "FLUX.1 [pro] Fill Fine-tuned is a high-performance endpoint for the FLUX.1 [pro] model with a fine-tuned LoRA that enables rapid transformation of existing images, delivering high-quality style transfers and image modifications with the core FLUX capabilities.",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/moondream-next/detection": {
        "class_name": "MoondreamNextDetection",
        "docstring": "MoonDreamNext Detection is a multimodal vision-language model for gaze detection, bbox detection, point detection, and more.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/bria/eraser": {
        "class_name": "BriaEraserV2",
        "docstring": "Bria Eraser enables precise removal of unwanted objects from images while maintaining high-quality outputs. Trained exclusively on licensed data for safe and risk-free commercial use. Access the model's source code and weights: https://bria.ai/contact-us",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/bria/expand": {
        "class_name": "BriaExpand",
        "docstring": "Bria Expand expands images beyond their borders in high quality. Trained exclusively on licensed data for safe and risk-free commercial use. Access the model's source code and weights: https://bria.ai/contact-us",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/bria/genfill": {
        "class_name": "BriaGenfill",
        "docstring": "Bria GenFill enables high-quality object addition or visual transformation. Trained exclusively on licensed data for safe and risk-free commercial use. Access the model's source code and weights: https://bria.ai/contact-us",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/bria/product-shot": {
        "class_name": "BriaProductShot",
        "docstring": "Place any product in any scenery with just a prompt or reference image while maintaining high integrity of the product. Trained exclusively on licensed data for safe and risk-free commercial use and optimized for eCommerce.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/bria/background/remove": {
        "class_name": "BriaBackgroundRemove",
        "docstring": "Bria RMBG 2.0 enables seamless removal of backgrounds from images, ideal for professional editing tasks. Trained exclusively on licensed data for safe and risk-free commercial use. Model weights for commercial use are available here: https://share-eu1.hsforms.com/2GLpEVQqJTI2Lj7AMYwgfIwf4e04",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/bria/background/replace": {
        "class_name": "BriaBackgroundReplaceV2",
        "docstring": "Bria Background Replace allows for efficient swapping of backgrounds in images via text prompts or reference image, delivering realistic and polished results. Trained exclusively on licensed data for safe and risk-free commercial use ",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/flux-lora-fill": {
        "class_name": "FluxLoraFill",
        "docstring": "FLUX.1 [dev] Fill is a high-performance endpoint for the FLUX.1 [pro] model that enables rapid transformation of existing images, delivering high-quality style transfers and image modifications with the core FLUX capabilities.",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/cat-vton": {
        "class_name": "CatVton",
        "docstring": "Image based high quality Virtual Try-On",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/leffa/pose-transfer": {
        "class_name": "LeffaPoseTransfer",
        "docstring": "Leffa Pose Transfer is an endpoint for changing pose of an image with a reference image.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/leffa/virtual-tryon": {
        "class_name": "LeffaVirtualTryon",
        "docstring": "Leffa Virtual TryOn is a high quality image based Try-On endpoint which can be used for commercial try on.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/ideogram/v2/turbo/edit": {
        "class_name": "IdeogramV2TurboEdit",
        "docstring": "Edit images faster with Ideogram V2 Turbo. Quick modifications and adjustments while preserving the high-quality standards and realistic outputs of Ideogram.",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "fast"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/ideogram/v2/turbo/remix": {
        "class_name": "IdeogramV2TurboRemix",
        "docstring": "Rapidly create image variations with Ideogram V2 Turbo Remix. Fast and efficient reimagining of existing images while maintaining creative control through prompt guidance.",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "fast"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/flux-pro/v1.1/redux": {
        "class_name": "FluxProV11Redux",
        "docstring": "FLUX1.1 [pro] Redux is a high-performance endpoint for the FLUX1.1 [pro] model that enables rapid transformation of existing images, delivering high-quality style transfers and image modifications with the core FLUX capabilities.",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/flux-lora-depth": {
        "class_name": "FluxLoraDepth",
        "docstring": "Generate high-quality images from depth maps using Flux.1 [dev] depth estimation model. The model produces accurate depth representations for scene understanding and 3D visualization.",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/flux-pro/v1.1-ultra/redux": {
        "class_name": "FluxProV11UltraRedux",
        "docstring": "FLUX1.1 [pro] ultra Redux is a high-performance endpoint for the FLUX1.1 [pro] model that enables rapid transformation of existing images, delivering high-quality style transfers and image modifications with the core FLUX capabilities.",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/iclight-v2": {
        "class_name": "IclightV2",
        "docstring": "An endpoint for re-lighting photos and changing their backgrounds per a given description",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/flux-differential-diffusion": {
        "class_name": "FluxDifferentialDiffusion",
        "docstring": "FLUX.1 Differential Diffusion is a rapid endpoint that enables swift, granular control over image transformations through change maps, delivering fast and precise region-specific modifications while maintaining FLUX.1 [dev]'s high-quality output.",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/flux-pulid": {
        "class_name": "FluxPulid",
        "docstring": "An endpoint for personalized image generation using Flux as per given description.",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/birefnet/v2": {
        "class_name": "BirefnetV2",
        "docstring": "bilateral reference framework (BiRefNet) for high-resolution dichotomous image segmentation (DIS)",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/live-portrait/image": {
        "class_name": "LivePortraitImage",
        "docstring": "Transfer expression from a video to a portrait.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/flux-general/rf-inversion": {
        "class_name": "FluxGeneralRfInversion",
        "docstring": "A general purpose endpoint for the FLUX.1 [dev] model, implementing the RF-Inversion pipeline. This can be used to edit a reference image based on a prompt.",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-preprocessors/hed": {
        "class_name": "ImagePreprocessorsHed",
        "docstring": "Holistically-Nested Edge Detection (HED) preprocessor.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-preprocessors/depth-anything/v2": {
        "class_name": "ImagePreprocessorsDepthAnythingV2",
        "docstring": "Depth Anything v2 preprocessor.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-preprocessors/scribble": {
        "class_name": "ImagePreprocessorsScribble",
        "docstring": "Scribble preprocessor.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-preprocessors/mlsd": {
        "class_name": "ImagePreprocessorsMlsd",
        "docstring": "M-LSD line segment detection preprocessor.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-preprocessors/sam": {
        "class_name": "ImagePreprocessorsSam",
        "docstring": "Segment Anything Model (SAM) preprocessor.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-preprocessors/midas": {
        "class_name": "ImagePreprocessorsMidas",
        "docstring": "MiDaS depth estimation preprocessor.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-preprocessors/teed": {
        "class_name": "ImagePreprocessorsTeed",
        "docstring": "TEED (Temporal Edge Enhancement Detection) preprocessor.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-preprocessors/lineart": {
        "class_name": "ImagePreprocessorsLineart",
        "docstring": "Line art preprocessor.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-preprocessors/zoe": {
        "class_name": "ImagePreprocessorsZoe",
        "docstring": "ZoeDepth preprocessor.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/image-preprocessors/pidi": {
        "class_name": "ImagePreprocessorsPidi",
        "docstring": "PIDI (Pidinet) preprocessor.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/sam2/image": {
        "class_name": "Sam2Image",
        "docstring": "SAM 2 is a model for segmenting images and videos in real-time.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/flux-general/inpainting": {
        "class_name": "FluxGeneralInpainting",
        "docstring": "FLUX General Inpainting is a versatile endpoint that enables precise image editing and completion, supporting multiple AI extensions including LoRA, ControlNet, and IP-Adapter for enhanced control over inpainting results and sophisticated image modifications.",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/flux-general/image-to-image": {
        "class_name": "FluxGeneralImageToImage",
        "docstring": "FLUX General Image-to-Image is a versatile endpoint that transforms existing images with support for LoRA, ControlNet, and IP-Adapter extensions, enabling precise control over style transfer, modifications, and artistic variations through multiple guidance methods.",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/flux-general/differential-diffusion": {
        "class_name": "FluxGeneralDifferentialDiffusion",
        "docstring": "A specialized FLUX endpoint combining differential diffusion control with LoRA, ControlNet, and IP-Adapter support, enabling precise, region-specific image transformations through customizable change maps.",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/flux-lora/image-to-image": {
        "class_name": "FluxLoraImageToImage",
        "docstring": "FLUX LoRA Image-to-Image is a high-performance endpoint that transforms existing images using FLUX models, leveraging LoRA adaptations to enable rapid and precise image style transfer, modifications, and artistic variations.",
        "tags": ["flux", "editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/sdxl-controlnet-union/inpainting": {
        "class_name": "SdxlControlnetUnionInpainting",
        "docstring": "An efficent SDXL multi-controlnet inpainting model.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/sdxl-controlnet-union/image-to-image": {
        "class_name": "SdxlControlnetUnionImageToImage",
        "docstring": "An efficent SDXL multi-controlnet image-to-image model.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/era-3d": {
        "class_name": "Era3d",
        "docstring": "A powerful image to novel multiview model with normals.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/florence-2-large/dense-region-caption": {
        "class_name": "Florence2LargeDenseRegionCaption",
        "docstring": "Florence-2 is an advanced vision foundation model that uses a prompt-based approach to handle a wide range of vision and vision-language tasks",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/florence-2-large/referring-expression-segmentation": {
        "class_name": "Florence2LargeReferringExpressionSegmentation",
        "docstring": "Florence-2 is an advanced vision foundation model that uses a prompt-based approach to handle a wide range of vision and vision-language tasks",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/florence-2-large/object-detection": {
        "class_name": "Florence2LargeObjectDetection",
        "docstring": "Florence-2 is an advanced vision foundation model that uses a prompt-based approach to handle a wide range of vision and vision-language tasks",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/florence-2-large/open-vocabulary-detection": {
        "class_name": "Florence2LargeOpenVocabularyDetection",
        "docstring": "Florence-2 is an advanced vision foundation model that uses a prompt-based approach to handle a wide range of vision and vision-language tasks",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/florence-2-large/caption-to-phrase-grounding": {
        "class_name": "Florence2LargeCaptionToPhraseGrounding",
        "docstring": "Florence-2 is an advanced vision foundation model that uses a prompt-based approach to handle a wide range of vision and vision-language tasks",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/florence-2-large/region-proposal": {
        "class_name": "Florence2LargeRegionProposal",
        "docstring": "Florence-2 is an advanced vision foundation model that uses a prompt-based approach to handle a wide range of vision and vision-language tasks",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/florence-2-large/ocr-with-region": {
        "class_name": "Florence2LargeOcrWithRegion",
        "docstring": "Florence-2 is an advanced vision foundation model that uses a prompt-based approach to handle a wide range of vision and vision-language tasks",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/florence-2-large/region-to-segmentation": {
        "class_name": "Florence2LargeRegionToSegmentation",
        "docstring": "Florence-2 is an advanced vision foundation model that uses a prompt-based approach to handle a wide range of vision and vision-language tasks",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/stable-diffusion-v3-medium/image-to-image": {
        "class_name": "StableDiffusionV3MediumImageToImage",
        "docstring": "Stable Diffusion 3 Medium (Image to Image) is a Multimodal Diffusion Transformer (MMDiT) model that improves image quality, typography, prompt understanding, and efficiency.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/dwpose": {
        "class_name": "Dwpose",
        "docstring": "Predict poses from images.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/sd15-depth-controlnet": {
        "class_name": "Sd15DepthControlnet",
        "docstring": "SD 1.5 ControlNet",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/omni-zero": {
        "class_name": "OmniZero",
        "docstring": "Any pose, any style, any identity",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/hyper-sdxl/image-to-image": {
        "class_name": "HyperSdxlImageToImage",
        "docstring": "Hyper-charge SDXL's performance and creativity.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/hyper-sdxl/inpainting": {
        "class_name": "HyperSdxlInpainting",
        "docstring": "Hyper-charge SDXL's performance and creativity.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/ip-adapter-face-id": {
        "class_name": "IpAdapterFaceId",
        "docstring": "High quality zero-shot personalization",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/lora/inpaint": {
        "class_name": "LoraInpaint",
        "docstring": "Run Any Stable Diffusion model with customizable LoRA weights.",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/lora/image-to-image": {
        "class_name": "LoraImageToImage",
        "docstring": "Run Any Stable Diffusion model with customizable LoRA weights.",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "lora"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/fast-sdxl/image-to-image": {
        "class_name": "FastSdxlImageToImage",
        "docstring": "Run SDXL at the speed of light",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "fast"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/fast-sdxl/inpainting": {
        "class_name": "FastSdxlInpainting",
        "docstring": "Run SDXL at the speed of light",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "fast"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/face-to-sticker": {
        "class_name": "FaceToSticker",
        "docstring": "Create stickers from faces.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/photomaker": {
        "class_name": "Photomaker",
        "docstring": "Customizing Realistic Human Photos via Stacked ID Embedding",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/creative-upscaler": {
        "class_name": "CreativeUpscaler",
        "docstring": "Create creative upscaled images.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/playground-v25/image-to-image": {
        "class_name": "PlaygroundV25ImageToImage",
        "docstring": "State-of-the-art open-source model in aesthetic quality",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/fast-lightning-sdxl/image-to-image": {
        "class_name": "FastLightningSdxlImageToImage",
        "docstring": "Run SDXL at the speed of light",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "fast"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/fast-lightning-sdxl/inpainting": {
        "class_name": "FastLightningSdxlInpainting",
        "docstring": "Run SDXL at the speed of light",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "fast"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/playground-v25/inpainting": {
        "class_name": "PlaygroundV25Inpainting",
        "docstring": "State-of-the-art open-source model in aesthetic quality",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/fast-lcm-diffusion/inpainting": {
        "class_name": "FastLcmDiffusionInpainting",
        "docstring": "Run SDXL at the speed of light",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "fast"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/fast-lcm-diffusion/image-to-image": {
        "class_name": "FastLcmDiffusionImageToImage",
        "docstring": "Run SDXL at the speed of light",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "fast"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/retoucher": {
        "class_name": "Retoucher",
        "docstring": "Automatically retouches faces to smooth skin and remove blemishes.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/imageutils/depth": {
        "class_name": "ImageutilsDepth",
        "docstring": "Create depth maps using Midas depth estimation.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/imageutils/marigold-depth": {
        "class_name": "ImageutilsMarigoldDepth",
        "docstring": "Create depth maps using Marigold depth estimation.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/pulid": {
        "class_name": "Pulid",
        "docstring": "Tuning-free ID customization.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/fast-sdxl-controlnet-canny/image-to-image": {
        "class_name": "FastSdxlControlnetCannyImageToImage",
        "docstring": "Generate Images with ControlNet.",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "fast"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/fast-sdxl-controlnet-canny/inpainting": {
        "class_name": "FastSdxlControlnetCannyInpainting",
        "docstring": "Generate Images with ControlNet.",
        "tags": ["editing", "transformation", "image-to-image", "img2img", "fast"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/lcm-sd15-i2i": {
        "class_name": "LcmSd15I2i",
        "docstring": "Produce high-quality images with minimal inference steps. Optimized for 512x512 input image size.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/inpaint": {
        "class_name": "Inpaint",
        "docstring": "Inpaint images with SD and SDXL",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/esrgan": {
        "class_name": "Esrgan",
        "docstring": "Upscale images by a given factor.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
    },
    "fal-ai/imageutils/rembg": {
        "class_name": "ImageutilsRembg",
        "docstring": "Remove the background from an image.",
        "tags": ["editing", "transformation", "image-to-image", "img2img"],
        "use_cases": [
            "Professional photo editing and enhancement",
            "Creative image transformations",
            "Batch image processing workflows",
            "Product photography refinement",
            "Automated image optimization",
        ],
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
