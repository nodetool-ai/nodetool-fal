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
    
    "fal-ai/amt-interpolation/frame-interpolation": {
        "class_name": "AMTFrameInterpolation",
        "docstring": "AMT Frame Interpolation creates smooth transitions between image frames.",
        "tags": ["video", "interpolation", "frame-generation", "amt", "image-to-video"],
        "use_cases": [
            "Create smooth transitions between images",
            "Generate intermediate frames",
            "Animate image sequences",
            "Create video from image pairs",
            "Produce smooth motion effects"
        ],
        "basic_fields": ["image"]
    },
    
    "fal-ai/ai-avatar": {
        "class_name": "AIAvatar",
        "docstring": "MultiTalk generates talking avatar videos from images and audio files.",
        "tags": ["video", "avatar", "talking-head", "multitalk", "image-to-video"],
        "use_cases": [
            "Create talking avatar videos",
            "Animate portrait photos with audio",
            "Generate spokesperson videos",
            "Produce avatar presentations",
            "Create personalized video messages"
        ],
        "basic_fields": ["image", "audio"]
    },
    
    "fal-ai/ai-avatar/single-text": {
        "class_name": "AIAvatarSingleText",
        "docstring": "MultiTalk generates talking avatar videos from an image and text input.",
        "tags": ["video", "avatar", "talking-head", "text-to-speech", "image-to-video"],
        "use_cases": [
            "Create avatar videos from text",
            "Generate talking heads with TTS",
            "Produce text-driven avatars",
            "Create virtual presenters",
            "Generate automated spokesperson videos"
        ],
        "basic_fields": ["image", "text"]
    },
    
    "fal-ai/ai-avatar/multi-text": {
        "class_name": "AIAvatarMultiText",
        "docstring": "MultiTalk generates multi-speaker avatar videos from images and text.",
        "tags": ["video", "avatar", "multi-speaker", "talking-head", "image-to-video"],
        "use_cases": [
            "Create multi-speaker conversations",
            "Generate dialogue between avatars",
            "Produce interactive presentations",
            "Create conversational content",
            "Generate multi-character scenes"
        ],
        "basic_fields": ["images", "texts"]
    },
    
    "fal-ai/ai-avatar/multi": {
        "class_name": "AIAvatarMulti",
        "docstring": "MultiTalk generates multi-speaker avatar videos with audio synchronization.",
        "tags": ["video", "avatar", "multi-speaker", "talking-head", "image-to-video"],
        "use_cases": [
            "Create multi-speaker videos with audio",
            "Generate synchronized dialogue",
            "Produce conversation videos",
            "Create interactive characters",
            "Generate multi-avatar content"
        ],
        "basic_fields": ["images", "audio"]
    },
    
    "fal-ai/bytedance/seedance/v1.5/pro/image-to-video": {
        "class_name": "SeeDanceV15ProImageToVideo",
        "docstring": "SeeDance v1.5 Pro generates high-quality dance videos from images.",
        "tags": ["video", "dance", "animation", "seedance", "bytedance", "image-to-video"],
        "use_cases": [
            "Animate photos into dance videos",
            "Create dance choreography from images",
            "Generate dance performances",
            "Produce music video content",
            "Create dance training materials"
        ],
        "basic_fields": ["image", "prompt"]
    },
    
    "fal-ai/bytedance/seedance/v1/pro/fast/image-to-video": {
        "class_name": "SeeDanceV1ProFastImageToVideo",
        "docstring": "SeeDance v1 Pro Fast generates dance videos quickly from images.",
        "tags": ["video", "dance", "fast", "seedance", "bytedance", "image-to-video"],
        "use_cases": [
            "Rapidly generate dance videos",
            "Quick dance animation",
            "Fast dance prototypes",
            "Create dance previews",
            "Efficient dance video generation"
        ],
        "basic_fields": ["image", "prompt"]
    },
    
    "fal-ai/bytedance/seedance/v1/lite/reference-to-video": {
        "class_name": "SeeDanceV1LiteReferenceToVideo",
        "docstring": "SeeDance v1 Lite generates lightweight dance videos using reference images.",
        "tags": ["video", "dance", "lite", "reference", "seedance", "image-to-video"],
        "use_cases": [
            "Generate efficient dance videos",
            "Create reference-based animations",
            "Produce lightweight dance content",
            "Generate quick dance outputs",
            "Create optimized dance videos"
        ],
        "basic_fields": ["image", "reference"]
    },
    
    "fal-ai/bytedance/video-stylize": {
        "class_name": "ByteDanceVideoStylize",
        "docstring": "ByteDance Video Stylize applies artistic styles to image-based video generation.",
        "tags": ["video", "style-transfer", "artistic", "bytedance", "image-to-video"],
        "use_cases": [
            "Apply artistic styles to videos",
            "Create stylized video content",
            "Generate artistic animations",
            "Produce style-transferred videos",
            "Create visually unique content"
        ],
        "basic_fields": ["image", "style"]
    },
    
    "fal-ai/bytedance/omnihuman/v1.5": {
        "class_name": "OmniHumanV15",
        "docstring": "OmniHuman v1.5 generates realistic human videos from images.",
        "tags": ["video", "human", "realistic", "bytedance", "image-to-video"],
        "use_cases": [
            "Generate realistic human videos",
            "Create human motion animations",
            "Produce lifelike character videos",
            "Generate human performances",
            "Create realistic human content"
        ],
        "basic_fields": ["image", "prompt"]
    },
    
    "fal-ai/cogvideox-5b/image-to-video": {
        "class_name": "CogVideoX5BImageToVideo",
        "docstring": "CogVideoX-5B generates high-quality videos from images with advanced motion.",
        "tags": ["video", "generation", "cogvideo", "image-to-video", "img2vid"],
        "use_cases": [
            "Generate videos from images",
            "Create dynamic image animations",
            "Produce high-quality video content",
            "Animate static images",
            "Generate motion from photos"
        ],
        "basic_fields": ["image", "prompt"]
    },
    
    "fal-ai/stable-video": {
        "class_name": "StableVideoImageToVideo",
        "docstring": "Stable Video generates consistent video animations from images.",
        "tags": ["video", "generation", "stable", "consistent", "image-to-video"],
        "use_cases": [
            "Generate stable video animations",
            "Create consistent motion",
            "Produce reliable video outputs",
            "Animate images consistently",
            "Generate predictable videos"
        ],
        "basic_fields": ["image", "prompt"]
    },
    
    "fal-ai/hunyuan-video/image-to-video": {
        "class_name": "HunyuanImageToVideo",
        "docstring": "Hunyuan Video generates high-quality videos from images with advanced AI.",
        "tags": ["video", "generation", "hunyuan", "tencent", "image-to-video"],
        "use_cases": [
            "Generate cinematic videos from images",
            "Create high-quality animations",
            "Produce professional video content",
            "Animate images with detail",
            "Generate advanced video effects"
        ],
        "basic_fields": ["image", "prompt"]
    },
    
    "fal-ai/ltx-video/image-to-video": {
        "class_name": "LTXImageToVideo",
        "docstring": "LTX Video generates temporally consistent videos from images.",
        "tags": ["video", "generation", "ltx", "temporal", "image-to-video"],
        "use_cases": [
            "Generate temporally consistent videos",
            "Create smooth image animations",
            "Produce coherent video sequences",
            "Animate with temporal awareness",
            "Generate fluid motion videos"
        ],
        "basic_fields": ["image", "prompt"]
    },
    
    "fal-ai/kling-video/v1/standard/image-to-video": {
        "class_name": "KlingVideoV1StandardImageToVideo",
        "docstring": "Kling Video v1 Standard generates videos from images with balanced quality.",
        "tags": ["video", "generation", "kling", "standard", "image-to-video"],
        "use_cases": [
            "Generate standard quality videos",
            "Create balanced video animations",
            "Produce efficient video content",
            "Generate videos for web use",
            "Create moderate quality outputs"
        ],
        "basic_fields": ["image", "prompt"]
    },
    
    "fal-ai/kling-video/v1/pro/image-to-video": {
        "class_name": "KlingVideoV1ProImageToVideo",
        "docstring": "Kling Video v1 Pro generates professional quality videos from images.",
        "tags": ["video", "generation", "kling", "pro", "professional", "image-to-video"],
        "use_cases": [
            "Generate professional videos",
            "Create high-quality animations",
            "Produce premium video content",
            "Generate cinematic outputs",
            "Create professional grade videos"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # Pixverse v5.6
    "fal-ai/pixverse/v5.6/transition": {
        "class_name": "PixverseV56Transition",
        "docstring": "Pixverse v5.6 Transition creates smooth video transitions between two images with professional effects.",
        "tags": ["video", "transition", "pixverse", "v5.6", "effects"],
        "use_cases": [
            "Create smooth transitions between images",
            "Generate professional video effects",
            "Produce seamless image morphing",
            "Create transition animations",
            "Generate video connecting two scenes"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # Vidu Q2
    "fal-ai/vidu/q2/reference-to-video/pro": {
        "class_name": "ViduQ2ReferenceToVideoPro",
        "docstring": "Vidu Q2 Reference-to-Video Pro generates professional quality videos using reference images for style and content.",
        "tags": ["video", "generation", "vidu", "q2", "pro", "reference"],
        "use_cases": [
            "Generate pro videos from references",
            "Create style-consistent animations",
            "Produce reference-guided videos",
            "Generate videos matching examples",
            "Create professional reference-based content"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # Wan v2.6 Flash
    "wan/v2.6/image-to-video/flash": {
        "class_name": "WanV26ImageToVideoFlash",
        "docstring": "Wan v2.6 Flash generates videos from images with ultra-fast processing for rapid iteration.",
        "tags": ["video", "generation", "wan", "v2.6", "flash", "fast"],
        "use_cases": [
            "Generate videos at maximum speed",
            "Create rapid video prototypes",
            "Produce instant video previews",
            "Generate quick video iterations",
            "Create fast video animations"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "wan/v2.6/image-to-video": {
        "class_name": "WanV26ImageToVideo",
        "docstring": "Wan v2.6 generates high-quality videos from images with balanced quality and performance.",
        "tags": ["video", "generation", "wan", "v2.6", "image-to-video"],
        "use_cases": [
            "Generate quality videos from images",
            "Create balanced video animations",
            "Produce reliable video content",
            "Generate consistent videos",
            "Create professional animations"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # LTX-2 19B Family
    "fal-ai/ltx-2-19b/image-to-video": {
        "class_name": "Ltx219BImageToVideo",
        "docstring": "LTX-2 19B generates high-quality videos from images using the powerful 19-billion parameter model.",
        "tags": ["video", "generation", "ltx-2", "19b", "large-model"],
        "use_cases": [
            "Generate high-quality videos with large model",
            "Create detailed video animations",
            "Produce superior video content",
            "Generate videos with powerful AI",
            "Create premium video animations"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "fal-ai/ltx-2-19b/image-to-video/lora": {
        "class_name": "Ltx219BImageToVideoLora",
        "docstring": "LTX-2 19B with LoRA enables custom-trained 19B models for specialized video generation.",
        "tags": ["video", "generation", "ltx-2", "19b", "lora", "custom"],
        "use_cases": [
            "Generate videos with custom 19B model",
            "Create specialized video content",
            "Produce domain-specific animations",
            "Generate with fine-tuned large model",
            "Create customized video animations"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "fal-ai/ltx-2-19b/distilled/image-to-video": {
        "class_name": "Ltx219BDistilledImageToVideo",
        "docstring": "LTX-2 19B Distilled generates videos efficiently using knowledge distillation from the 19B model.",
        "tags": ["video", "generation", "ltx-2", "19b", "distilled", "efficient"],
        "use_cases": [
            "Generate videos efficiently with distilled model",
            "Create fast quality video animations",
            "Produce optimized video content",
            "Generate videos with good performance",
            "Create balanced quality-speed videos"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "fal-ai/ltx-2-19b/distilled/image-to-video/lora": {
        "class_name": "Ltx219BDistilledImageToVideoLora",
        "docstring": "LTX-2 19B Distilled with LoRA combines efficient generation with custom-trained models.",
        "tags": ["video", "generation", "ltx-2", "19b", "distilled", "lora"],
        "use_cases": [
            "Generate videos with custom distilled model",
            "Create efficient specialized content",
            "Produce fast domain-specific videos",
            "Generate with optimized custom model",
            "Create quick customized animations"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # Wan Move
    "fal-ai/wan-move": {
        "class_name": "WanMove",
        "docstring": "Wan Move generates videos with natural motion and movement from static images.",
        "tags": ["video", "generation", "wan", "motion", "animation"],
        "use_cases": [
            "Add natural motion to images",
            "Create animated movements",
            "Produce dynamic video content",
            "Generate moving scenes from stills",
            "Create motion animations"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # Kandinsky5 Pro
    "fal-ai/kandinsky5-pro/image-to-video": {
        "class_name": "Kandinsky5ProImageToVideo",
        "docstring": "Kandinsky5 Pro generates professional quality videos from images with artistic style and control.",
        "tags": ["video", "generation", "kandinsky", "pro", "artistic"],
        "use_cases": [
            "Generate artistic videos from images",
            "Create stylized video animations",
            "Produce creative video content",
            "Generate videos with artistic flair",
            "Create professional artistic videos"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # Live Avatar
    "fal-ai/live-avatar": {
        "class_name": "LiveAvatar",
        "docstring": "Live Avatar creates animated talking avatars from portrait images with realistic lip-sync and expressions.",
        "tags": ["video", "avatar", "talking-head", "animation", "portrait"],
        "use_cases": [
            "Create talking avatar videos",
            "Animate portrait images",
            "Generate lip-synced avatars",
            "Produce speaking character videos",
            "Create animated presenters"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # Hunyuan Video v1.5
    "fal-ai/hunyuan-video-v1.5/image-to-video": {
        "class_name": "HunyuanVideoV15ImageToVideo",
        "docstring": "Hunyuan Video v1.5 generates high-quality videos from images with advanced AI capabilities.",
        "tags": ["video", "generation", "hunyuan", "v1.5", "advanced"],
        "use_cases": [
            "Generate advanced quality videos",
            "Create sophisticated animations",
            "Produce high-fidelity video content",
            "Generate videos with AI excellence",
            "Create cutting-edge video animations"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # Kling Video O1
    "fal-ai/kling-video/o1/standard/image-to-video": {
        "class_name": "KlingVideoO1StandardImageToVideo",
        "docstring": "Kling Video O1 Standard generates videos with optimized standard quality from images.",
        "tags": ["video", "generation", "kling", "o1", "standard"],
        "use_cases": [
            "Generate standard O1 quality videos",
            "Create optimized video animations",
            "Produce efficient video content",
            "Generate balanced quality videos",
            "Create standard tier animations"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "fal-ai/kling-video/o1/standard/reference-to-video": {
        "class_name": "KlingVideoO1StandardReferenceToVideo",
        "docstring": "Kling Video O1 Standard generates videos using reference images for style consistency.",
        "tags": ["video", "generation", "kling", "o1", "standard", "reference"],
        "use_cases": [
            "Generate videos from reference images",
            "Create style-consistent animations",
            "Produce reference-guided content",
            "Generate videos matching examples",
            "Create standardized reference videos"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # Kling Video v2.6 Pro
    "fal-ai/kling-video/v2.6/pro/image-to-video": {
        "class_name": "KlingVideoV26ProImageToVideo",
        "docstring": "Kling Video v2.6 Pro generates professional quality videos with latest model improvements.",
        "tags": ["video", "generation", "kling", "v2.6", "pro"],
        "use_cases": [
            "Generate professional v2.6 videos",
            "Create latest quality animations",
            "Produce premium video content",
            "Generate advanced videos",
            "Create pro-tier animations"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # Kling Video AI Avatar v2
    "fal-ai/kling-video/ai-avatar/v2/standard": {
        "class_name": "KlingVideoAiAvatarV2Standard",
        "docstring": "Kling Video AI Avatar v2 Standard creates animated talking avatars with standard quality.",
        "tags": ["video", "avatar", "kling", "v2", "standard", "talking-head"],
        "use_cases": [
            "Create standard quality talking avatars",
            "Animate portraits with speech",
            "Generate avatar presentations",
            "Produce speaking character videos",
            "Create AI-driven avatars"
        ],
        "basic_fields": ["image", "prompt"]
    },

    "fal-ai/kling-video/ai-avatar/v2/pro": {
        "class_name": "KlingVideoAiAvatarV2Pro",
        "docstring": "Kling Video AI Avatar v2 Pro creates professional quality animated talking avatars with enhanced realism.",
        "tags": ["video", "avatar", "kling", "v2", "pro", "talking-head"],
        "use_cases": [
            "Create professional talking avatars",
            "Animate portraits with high quality",
            "Generate realistic avatar videos",
            "Produce premium speaking characters",
            "Create pro-grade AI avatars"
        ],
        "basic_fields": ["image", "prompt"]
    },

    # Creatify Aurora
    "fal-ai/creatify/aurora": {
        "class_name": "CreatifyAurora",
        "docstring": "Creatify Aurora generates creative and visually stunning videos from images with unique effects.",
        "tags": ["video", "generation", "creatify", "aurora", "creative", "effects"],
        "use_cases": [
            "Generate creative visual effects videos",
            "Create stunning video animations",
            "Produce artistic video content",
            "Generate unique video effects",
            "Create visually impressive videos"
        ],
        "basic_fields": ["image", "prompt"]
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
