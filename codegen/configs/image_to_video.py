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
