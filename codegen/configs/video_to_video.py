"""
Configuration for video_to_video module.

This config file defines overrides and customizations for video-to-video nodes.
"""

from typing import Any


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
    "fal-ai/amt-interpolation": {
        "class_name": "AMTInterpolation",
        "docstring": "AMT (Any-to-Many Temporal) Interpolation creates smooth transitions between video frames.",
        "tags": ["video", "interpolation", "frame-generation", "amt", "video-to-video"],
        "use_cases": [
            "Increase video frame rate smoothly",
            "Create slow-motion effects",
            "Smooth out choppy video",
            "Generate intermediate frames",
            "Enhance video playback quality"
        ],
        "basic_fields": ["video"]
    },
    
    "half-moon-ai/ai-face-swap/faceswapvideo": {
        "class_name": "AIFaceSwapVideo",
        "docstring": "AI Face Swap replaces faces in videos with target faces while preserving expressions and movements.",
        "tags": ["video", "face-swap", "deepfake", "face-replacement", "video-to-video"],
        "use_cases": [
            "Replace faces in video content",
            "Create personalized video content",
            "Swap actors in video scenes",
            "Generate face replacement effects",
            "Create video with different faces"
        ],
        "basic_fields": ["video", "target_face"]
    },
    
    "fal-ai/fast-animatediff/video-to-video": {
        "class_name": "AnimateDiffVideoToVideo",
        "docstring": "AnimateDiff re-animates videos with new styles and effects using diffusion models.",
        "tags": ["video", "style-transfer", "animatediff", "re-animation", "video-to-video"],
        "use_cases": [
            "Restyle existing videos",
            "Apply artistic effects to videos",
            "Transform video aesthetics",
            "Create stylized video versions",
            "Generate video variations"
        ],
        "basic_fields": ["video", "prompt"]
    },
    
    "fal-ai/fast-animatediff/turbo/video-to-video": {
        "class_name": "AnimateDiffTurboVideoToVideo",
        "docstring": "AnimateDiff Turbo re-animates videos quickly with reduced generation time.",
        "tags": ["video", "style-transfer", "animatediff", "turbo", "fast", "video-to-video"],
        "use_cases": [
            "Quickly restyle videos",
            "Rapid video transformations",
            "Fast video effect application",
            "Efficient video processing",
            "Real-time video styling"
        ],
        "basic_fields": ["video", "prompt"]
    },
    
    "fal-ai/auto-caption": {
        "class_name": "AutoCaption",
        "docstring": "Auto Caption automatically generates and adds captions to videos with speech recognition.",
        "tags": ["video", "captions", "subtitles", "speech-to-text", "video-to-video"],
        "use_cases": [
            "Add subtitles to videos automatically",
            "Generate captions for accessibility",
            "Create multilingual subtitles",
            "Transcribe video speech",
            "Add text overlays to videos"
        ],
        "basic_fields": ["video"]
    },
    
    "fal-ai/ben/v2/video": {
        "class_name": "BenV2Video",
        "docstring": "Ben v2 Video enhances and processes video content with advanced AI techniques.",
        "tags": ["video", "enhancement", "processing", "ben", "video-to-video"],
        "use_cases": [
            "Enhance video quality",
            "Process video content",
            "Improve video clarity",
            "Apply video enhancements",
            "Optimize video output"
        ],
        "basic_fields": ["video"]
    },
    
    "fal-ai/birefnet/v2/video": {
        "class_name": "BiRefNetV2Video",
        "docstring": "BiRefNet v2 Video performs background removal from videos with high accuracy.",
        "tags": ["video", "background-removal", "segmentation", "birefnet", "video-to-video"],
        "use_cases": [
            "Remove backgrounds from videos",
            "Create transparent video backgrounds",
            "Isolate video subjects",
            "Generate video mattes",
            "Prepare videos for compositing"
        ],
        "basic_fields": ["video"]
    },
    
    "bria/bria_video_eraser/erase/mask": {
        "class_name": "BriaVideoEraserMask",
        "docstring": "Bria Video Eraser removes objects from videos using mask-based selection.",
        "tags": ["video", "object-removal", "eraser", "inpainting", "bria", "video-to-video"],
        "use_cases": [
            "Remove unwanted objects from videos",
            "Erase people or items from footage",
            "Clean up video backgrounds",
            "Remove watermarks from videos",
            "Edit video content seamlessly"
        ],
        "basic_fields": ["video", "mask"]
    },
    
    "bria/bria_video_eraser/erase/keypoints": {
        "class_name": "BriaVideoEraserKeypoints",
        "docstring": "Bria Video Eraser removes objects from videos using keypoint-based selection.",
        "tags": ["video", "object-removal", "eraser", "keypoints", "bria", "video-to-video"],
        "use_cases": [
            "Remove objects using keypoint selection",
            "Erase specific areas from videos",
            "Targeted video content removal",
            "Precision video editing",
            "Remove elements with point markers"
        ],
        "basic_fields": ["video", "keypoints"]
    },
    
    "bria/bria_video_eraser/erase/prompt": {
        "class_name": "BriaVideoEraserPrompt",
        "docstring": "Bria Video Eraser removes objects from videos using text prompt descriptions.",
        "tags": ["video", "object-removal", "eraser", "prompt", "bria", "video-to-video"],
        "use_cases": [
            "Remove objects by describing them",
            "Text-based video editing",
            "Natural language video cleanup",
            "Prompt-driven object removal",
            "Semantic video editing"
        ],
        "basic_fields": ["video", "prompt"]
    },
    
    "fal-ai/cogvideox-5b/video-to-video": {
        "class_name": "CogVideoX5BVideoToVideo",
        "docstring": "CogVideoX-5B transforms existing videos with new styles and effects.",
        "tags": ["video", "transformation", "cogvideo", "style-transfer", "video-to-video"],
        "use_cases": [
            "Transform video styles",
            "Apply effects to existing videos",
            "Restyle video content",
            "Generate video variations",
            "Create artistic video versions"
        ],
        "basic_fields": ["video", "prompt"]
    },
    
    "fal-ai/frame-forge": {
        "class_name": "FrameForge",
        "docstring": "Frame Forge processes and transforms individual video frames for creative effects.",
        "tags": ["video", "frame-processing", "effects", "transformation", "video-to-video"],
        "use_cases": [
            "Process video frames individually",
            "Apply per-frame effects",
            "Transform frame sequences",
            "Create frame-based effects",
            "Generate processed video output"
        ],
        "basic_fields": ["video"]
    },
    
    "fal-ai/hunyuan-video/video-to-video": {
        "class_name": "HunyuanVideoToVideo",
        "docstring": "Hunyuan Video transforms existing videos with advanced AI-powered effects.",
        "tags": ["video", "transformation", "hunyuan", "video-to-video"],
        "use_cases": [
            "Transform video content",
            "Apply AI effects to videos",
            "Restyle existing footage",
            "Generate video variations",
            "Create enhanced video versions"
        ],
        "basic_fields": ["video", "prompt"]
    },
    
    "fal-ai/ltx-video/video-to-video": {
        "class_name": "LTXVideoToVideo",
        "docstring": "LTX Video transforms videos with temporal consistency and high quality.",
        "tags": ["video", "transformation", "ltx", "temporal", "video-to-video"],
        "use_cases": [
            "Transform videos with consistency",
            "Apply temporal effects",
            "Generate smooth video transitions",
            "Create consistent video variations",
            "Maintain temporal coherence"
        ],
        "basic_fields": ["video", "prompt"]
    },
    
    "fal-ai/luma-dream-machine/video-to-video": {
        "class_name": "LumaDreamMachineVideoToVideo",
        "docstring": "Luma Dream Machine transforms videos with dreamlike artistic effects.",
        "tags": ["video", "transformation", "luma", "dream-machine", "artistic", "video-to-video"],
        "use_cases": [
            "Create dreamlike video effects",
            "Transform videos artistically",
            "Generate surreal video versions",
            "Apply creative effects",
            "Produce artistic video content"
        ],
        "basic_fields": ["video", "prompt"]
    },
    
    "fal-ai/minimax-video/v1/video-to-video": {
        "class_name": "MinimaxVideoV1VideoToVideo",
        "docstring": "Minimax Video v1 transforms videos efficiently with minimal resource usage.",
        "tags": ["video", "transformation", "minimax", "efficient", "video-to-video"],
        "use_cases": [
            "Transform videos efficiently",
            "Process videos with minimal resources",
            "Generate optimized video outputs",
            "Create scalable video transformations",
            "Efficient video processing"
        ],
        "basic_fields": ["video", "prompt"]
    },
    
    "fal-ai/runway-gen3/turbo/video-to-video": {
        "class_name": "RunwayGen3TurboVideoToVideo",
        "docstring": "Runway Gen-3 Turbo transforms videos quickly with high-quality output.",
        "tags": ["video", "transformation", "runway", "gen3", "turbo", "video-to-video"],
        "use_cases": [
            "Transform videos rapidly",
            "Quick video style transfers",
            "Fast video processing",
            "Real-time video effects",
            "Efficient video transformations"
        ],
        "basic_fields": ["video", "prompt"]
    },
    
    "fal-ai/stable-video/video-to-video": {
        "class_name": "StableVideoToVideo",
        "docstring": "Stable Video transforms videos with consistent and stable results.",
        "tags": ["video", "transformation", "stable", "consistent", "video-to-video"],
        "use_cases": [
            "Transform videos consistently",
            "Generate stable video outputs",
            "Create predictable video effects",
            "Maintain video stability",
            "Reliable video transformations"
        ],
        "basic_fields": ["video", "prompt"]
    },
    
    "fal-ai/wan-x-labs/svd-v1": {
        "class_name": "WanXLabsSVDV1",
        "docstring": "Wan X Labs SVD v1 performs stable video diffusion for video transformation.",
        "tags": ["video", "diffusion", "svd", "transformation", "video-to-video"],
        "use_cases": [
            "Apply diffusion effects to videos",
            "Transform videos with SVD",
            "Generate diffusion-based variations",
            "Create stable video transformations",
            "Produce diffusion video effects"
        ],
        "basic_fields": ["video"]
    },
    
    "fal-ai/video-upscaler": {
        "class_name": "VideoUpscaler",
        "docstring": "Video Upscaler enhances video resolution and quality using AI.",
        "tags": ["video", "upscaling", "enhancement", "resolution", "video-to-video"],
        "use_cases": [
            "Upscale low resolution videos",
            "Enhance video quality",
            "Increase video resolution",
            "Improve video clarity",
            "Restore old video footage"
        ],
        "basic_fields": ["video"]
    },
    
    "fal-ai/ccsr": {
        "class_name": "CCSR",
        "docstring": "CCSR (Controllable Color Style Restoration) restores and enhances video colors.",
        "tags": ["video", "color-restoration", "enhancement", "ccsr", "video-to-video"],
        "use_cases": [
            "Restore video colors",
            "Enhance video color quality",
            "Fix color issues in videos",
            "Improve video color grading",
            "Restore faded video footage"
        ],
        "basic_fields": ["video"]
    },
    
    "fal-ai/luma-photon/video-to-video": {
        "class_name": "LumaPhotonVideoToVideo",
        "docstring": "Luma Photon transforms videos with photorealistic effects and enhancements.",
        "tags": ["video", "transformation", "luma", "photon", "photorealistic", "video-to-video"],
        "use_cases": [
            "Create photorealistic video effects",
            "Transform videos realistically",
            "Generate realistic video variations",
            "Enhance video realism",
            "Produce lifelike video content"
        ],
        "basic_fields": ["video", "prompt"]
    },
    
    "fal-ai/kling-video/v1/standard/video-to-video": {
        "class_name": "KlingVideoV1StandardVideoToVideo",
        "docstring": "Kling Video v1 Standard transforms videos with balanced quality and speed.",
        "tags": ["video", "transformation", "kling", "standard", "video-to-video"],
        "use_cases": [
            "Transform videos efficiently",
            "Balanced video processing",
            "Standard quality transformations",
            "General purpose video effects",
            "Moderate speed processing"
        ],
        "basic_fields": ["video", "prompt"]
    },
    
    "fal-ai/kling-video/v1/pro/video-to-video": {
        "class_name": "KlingVideoV1ProVideoToVideo",
        "docstring": "Kling Video v1 Pro transforms videos with professional quality output.",
        "tags": ["video", "transformation", "kling", "pro", "professional", "video-to-video"],
        "use_cases": [
            "Professional video transformations",
            "High-quality video effects",
            "Premium video processing",
            "Cinematic video enhancements",
            "Professional grade output"
        ],
        "basic_fields": ["video", "prompt"]
    },
    
    "fal-ai/moondream/video": {
        "class_name": "MoondreamVideo",
        "docstring": "Moondream Video analyzes and processes video content with AI understanding.",
        "tags": ["video", "analysis", "understanding", "moondream", "video-to-video"],
        "use_cases": [
            "Analyze video content",
            "Process videos with AI understanding",
            "Extract video insights",
            "Generate video descriptions",
            "Intelligent video processing"
        ],
        "basic_fields": ["video"]
    },
    
    "fal-ai/video-depth-crafter": {
        "class_name": "VideoDepthCrafter",
        "docstring": "Video Depth Crafter generates depth maps from videos for 3D effects.",
        "tags": ["video", "depth-estimation", "3d", "depth-map", "video-to-video"],
        "use_cases": [
            "Generate depth maps from videos",
            "Create 3D effects from videos",
            "Extract depth information",
            "Enable video 3D conversion",
            "Produce depth-aware video effects"
        ],
        "basic_fields": ["video"]
    },
    
    "fal-ai/video-portrait": {
        "class_name": "VideoPortrait",
        "docstring": "Video Portrait processes and enhances portrait videos with face-aware effects.",
        "tags": ["video", "portrait", "face-processing", "enhancement", "video-to-video"],
        "use_cases": [
            "Process portrait videos",
            "Enhance face quality in videos",
            "Apply portrait effects",
            "Improve video selfies",
            "Face-aware video processing"
        ],
        "basic_fields": ["video"]
    },
    
    "fal-ai/viggle/v2": {
        "class_name": "ViggleV2",
        "docstring": "Viggle v2 applies motion and animation effects to video content.",
        "tags": ["video", "motion", "animation", "viggle", "video-to-video"],
        "use_cases": [
            "Apply motion effects to videos",
            "Animate video content",
            "Create dynamic video effects",
            "Generate motion-based variations",
            "Add movement to videos"
        ],
        "basic_fields": ["video"]
    },
    
    "fal-ai/video-retalking": {
        "class_name": "VideoRetalking",
        "docstring": "Video Retalking synchronizes lip movements in videos with new audio.",
        "tags": ["video", "lip-sync", "audio-sync", "retalking", "video-to-video"],
        "use_cases": [
            "Sync lips with new audio",
            "Dub videos naturally",
            "Change video dialogue",
            "Create multilingual videos",
            "Resync video speech"
        ],
        "basic_fields": ["video", "audio"]
    },
    
    "fal-ai/video-stabilizer": {
        "class_name": "VideoStabilizer",
        "docstring": "Video Stabilizer removes camera shake and stabilizes shaky video footage.",
        "tags": ["video", "stabilization", "shake-removal", "smoothing", "video-to-video"],
        "use_cases": [
            "Stabilize shaky videos",
            "Remove camera shake",
            "Smooth handheld footage",
            "Fix unstable video",
            "Improve video stability"
        ],
        "basic_fields": ["video"]
    },

    "fal-ai/ltx-2-19b/distilled/video-to-video/lora": {
        "class_name": "Ltx219BDistilledVideoToVideoLora",
        "docstring": "LTX-2 19B Distilled",
        "tags": ["video", "editing", "video-to-video", "vid2vid", "lora"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/ltx-2-19b/distilled/video-to-video": {
        "class_name": "Ltx219BDistilledVideoToVideo",
        "docstring": "LTX-2 19B Distilled",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/ltx-2-19b/video-to-video/lora": {
        "class_name": "Ltx219BVideoToVideoLora",
        "docstring": "LTX-2 19B",
        "tags": ["video", "editing", "video-to-video", "vid2vid", "lora"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/ltx-2-19b/video-to-video": {
        "class_name": "Ltx219BVideoToVideo",
        "docstring": "LTX-2 19B",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/ltx-2-19b/distilled/extend-video/lora": {
        "class_name": "Ltx219BDistilledExtendVideoLora",
        "docstring": "LTX-2 19B Distilled",
        "tags": ["video", "editing", "video-to-video", "vid2vid", "lora"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/ltx-2-19b/distilled/extend-video": {
        "class_name": "Ltx219BDistilledExtendVideo",
        "docstring": "LTX-2 19B Distilled",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/ltx-2-19b/extend-video/lora": {
        "class_name": "Ltx219BExtendVideoLora",
        "docstring": "LTX-2 19B",
        "tags": ["video", "editing", "video-to-video", "vid2vid", "lora"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/ltx-2-19b/extend-video": {
        "class_name": "Ltx219BExtendVideo",
        "docstring": "LTX-2 19B",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "bria/video/erase/keypoints": {
        "class_name": "BriaVideoEraseKeypoints",
        "docstring": "Video",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "bria/video/erase/prompt": {
        "class_name": "BriaVideoErasePrompt",
        "docstring": "Video",
        "tags": ["video", "editing", "video-to-video", "vid2vid", "professional"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "bria/video/erase/mask": {
        "class_name": "BriaVideoEraseMask",
        "docstring": "Video",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/lightx/relight": {
        "class_name": "LightxRelight",
        "docstring": "Lightx",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/lightx/recamera": {
        "class_name": "LightxRecamera",
        "docstring": "Lightx",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/kling-video/v2.6/standard/motion-control": {
        "class_name": "KlingVideoV2.6StandardMotionControl",
        "docstring": "Kling Video v2.6 Motion Control [Standard]",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/kling-video/v2.6/pro/motion-control": {
        "class_name": "KlingVideoV2.6ProMotionControl",
        "docstring": "Kling Video v2.6 Motion Control [Pro]",
        "tags": ["video", "editing", "video-to-video", "vid2vid", "professional"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "decart/lucy-restyle": {
        "class_name": "DecartLucyRestyle",
        "docstring": "Lucy Restyle",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/scail": {
        "class_name": "Scail",
        "docstring": "Scail",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "clarityai/crystal-video-upscaler": {
        "class_name": "ClarityaiCrystalVideoUpscaler",
        "docstring": "Crystal Upscaler [Video]",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "wan/v2.6/reference-to-video": {
        "class_name": "WanV2.6ReferenceToVideo",
        "docstring": "Wan v2.6 Reference to Video",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/veo3.1/fast/extend-video": {
        "class_name": "Veo3.1FastExtendVideo",
        "docstring": "Veo 3.1 Fast",
        "tags": ["video", "editing", "video-to-video", "vid2vid", "fast"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/veo3.1/extend-video": {
        "class_name": "Veo3.1ExtendVideo",
        "docstring": "Veo 3.1",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/kling-video/o1/standard/video-to-video/reference": {
        "class_name": "KlingVideoO1StandardVideoToVideoReference",
        "docstring": "Kling O1 Reference Video to Video [Standard]",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/kling-video/o1/standard/video-to-video/edit": {
        "class_name": "KlingVideoO1StandardVideoToVideoEdit",
        "docstring": "Kling O1 Edit Video [Standard]",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/steady-dancer": {
        "class_name": "SteadyDancer",
        "docstring": "Steady Dancer",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/one-to-all-animation/1.3b": {
        "class_name": "OneToAllAnimation1.3B",
        "docstring": "One To All Animation",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/one-to-all-animation/14b": {
        "class_name": "OneToAllAnimation14B",
        "docstring": "One To All Animation",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/wan-vision-enhancer": {
        "class_name": "WanVisionEnhancer",
        "docstring": "Wan Vision Enhancer",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/sync-lipsync/react-1": {
        "class_name": "SyncLipsyncReact1",
        "docstring": "Sync React-1",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "veed/video-background-removal/fast": {
        "class_name": "VeedVideoBackgroundRemovalFast",
        "docstring": "Video Background Removal",
        "tags": ["video", "editing", "video-to-video", "vid2vid", "fast"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/kling-video/o1/video-to-video/edit": {
        "class_name": "KlingVideoO1VideoToVideoEdit",
        "docstring": "Kling O1 Edit Video [Pro]",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/kling-video/o1/video-to-video/reference": {
        "class_name": "KlingVideoO1VideoToVideoReference",
        "docstring": "Kling O1 Reference Video to Video [Pro]",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "veed/video-background-removal": {
        "class_name": "VeedVideoBackgroundRemoval",
        "docstring": "Video Background Removal",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "veed/video-background-removal/green-screen": {
        "class_name": "VeedVideoBackgroundRemovalGreenScreen",
        "docstring": "Video Background Removal",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/ltx-2/retake-video": {
        "class_name": "Ltx2RetakeVideo",
        "docstring": "LTX Video 2.0 Retake",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "decart/lucy-edit/fast": {
        "class_name": "DecartLucyEditFast",
        "docstring": "Lucy Edit [Fast]",
        "tags": ["video", "editing", "video-to-video", "vid2vid", "fast"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/sam-3/video-rle": {
        "class_name": "Sam3VideoRle",
        "docstring": "Sam 3",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/sam-3/video": {
        "class_name": "Sam3Video",
        "docstring": "Sam 3",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/editto": {
        "class_name": "Editto",
        "docstring": "Editto",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/flashvsr/upscale/video": {
        "class_name": "FlashvsrUpscaleVideo",
        "docstring": "Flashvsr",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/workflow-utilities/auto-subtitle": {
        "class_name": "WorkflowUtilitiesAutoSubtitle",
        "docstring": "Workflow Utilities",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/bytedance-upscaler/upscale/video": {
        "class_name": "BytedanceUpscalerUpscaleVideo",
        "docstring": "Bytedance Upscaler",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/video-as-prompt": {
        "class_name": "VideoAsPrompt",
        "docstring": "Video As Prompt",
        "tags": ["video", "editing", "video-to-video", "vid2vid", "professional"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/vidu/q2/video-extension/pro": {
        "class_name": "ViduQ2VideoExtensionPro",
        "docstring": "Vidu",
        "tags": ["video", "editing", "video-to-video", "vid2vid", "professional"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "mirelo-ai/sfx-v1.5/video-to-video": {
        "class_name": "MireloAiSfxV1.5VideoToVideo",
        "docstring": "Mirelo SFX V1.5",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/krea-wan-14b/video-to-video": {
        "class_name": "KreaWan14BVideoToVideo",
        "docstring": "Krea Wan 14B",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/sora-2/video-to-video/remix": {
        "class_name": "Sora2VideoToVideoRemix",
        "docstring": "Sora 2",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/wan-vace-apps/long-reframe": {
        "class_name": "WanVaceAppsLongReframe",
        "docstring": "Wan 2.1 VACE Long Reframe",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/infinitalk/video-to-video": {
        "class_name": "InfinitalkVideoToVideo",
        "docstring": "Infinitalk",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/seedvr/upscale/video": {
        "class_name": "SeedvrUpscaleVideo",
        "docstring": "SeedVR2",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
    "fal-ai/wan-vace-apps/video-edit": {
        "class_name": "WanVaceAppsVideoEdit",
        "docstring": "Wan VACE Video Edit",
        "tags": ["video", "editing", "video-to-video", "vid2vid"],
        "use_cases": [
            "Video style transfer",
            "Video enhancement and restoration",
            "Automated video editing",
            "Special effects generation",
            "Content repurposing"

        ],
    },
}


def get_config(endpoint_id: str) -> dict[str, Any]:
    """Get config for an endpoint."""
    return CONFIGS.get(endpoint_id, {})
