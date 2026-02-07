"""
Configuration for text_to_video module.

This config file defines overrides and customizations for text-to-video nodes.
"""

from typing import Any


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
    "fal-ai/hunyuan-video": {
        "class_name": "HunyuanVideo",
        "docstring": "Hunyuan Video is Tencent's advanced text-to-video model for high-quality video generation.",
        "tags": ["video", "generation", "hunyuan", "text-to-video", "txt2vid"],
        "use_cases": [
            "Generate cinematic videos from text descriptions",
            "Create marketing videos from product descriptions",
            "Produce educational video content",
            "Generate creative video concepts",
            "Create animated scenes from stories"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/cogvideox-5b": {
        "class_name": "CogVideoX5B",
        "docstring": "CogVideoX-5B is a powerful open-source text-to-video generation model with 5 billion parameters.",
        "tags": ["video", "generation", "cogvideo", "text-to-video", "txt2vid"],
        "use_cases": [
            "Generate detailed videos from text prompts",
            "Create animated storytelling content",
            "Produce concept videos for pitches",
            "Generate video storyboards",
            "Create educational demonstrations"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/fast-animatediff/text-to-video": {
        "class_name": "AnimateDiffTextToVideo",
        "docstring": "AnimateDiff generates smooth animations from text prompts using diffusion models.",
        "tags": ["video", "generation", "animatediff", "animation", "text-to-video", "txt2vid"],
        "use_cases": [
            "Animate ideas from text descriptions",
            "Create animated content quickly",
            "Generate motion graphics from prompts",
            "Produce animated concept art",
            "Create video loops and sequences"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/fast-animatediff/turbo/text-to-video": {
        "class_name": "AnimateDiffTurboTextToVideo",
        "docstring": "AnimateDiff Turbo generates animations at lightning speed with reduced steps.",
        "tags": ["video", "generation", "animatediff", "turbo", "fast", "text-to-video", "txt2vid"],
        "use_cases": [
            "Rapidly prototype video animations",
            "Create quick video previews",
            "Generate animations with minimal latency",
            "Iterate on video concepts quickly",
            "Produce real-time animation effects"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/animatediff-sparsectrl-lcm": {
        "class_name": "AnimateDiffSparseCtrlLCM",
        "docstring": "AnimateDiff SparseCtrl LCM animates drawings with latent consistency models for fast generation.",
        "tags": ["video", "generation", "animatediff", "sparsectrl", "lcm", "animation", "text-to-video"],
        "use_cases": [
            "Animate hand-drawn sketches",
            "Bring drawings to life",
            "Create animated illustrations",
            "Generate animations from concept art",
            "Produce animation from sparse frames"
        ],
        "basic_fields": ["prompt"]
    },
    
    "veed/avatars/text-to-video": {
        "class_name": "VeedAvatarsTextToVideo",
        "docstring": "VEED Avatars generates talking avatar videos from text using realistic AI-powered characters.",
        "tags": ["video", "generation", "avatar", "talking-head", "veed", "text-to-video"],
        "use_cases": [
            "Create talking avatar presentations",
            "Generate spokesperson videos",
            "Produce educational talking head videos",
            "Create personalized video messages",
            "Generate multilingual avatar content"
        ],
        "basic_fields": ["prompt"]
    },
    
    "argil/avatars/text-to-video": {
        "class_name": "ArgilAvatarsTextToVideo",
        "docstring": "Argil Avatars creates realistic talking avatar videos from text descriptions.",
        "tags": ["video", "generation", "avatar", "talking-head", "argil", "text-to-video"],
        "use_cases": [
            "Generate avatar spokesperson videos",
            "Create virtual presenter content",
            "Produce automated video announcements",
            "Generate character-based narratives",
            "Create social media avatar videos"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/bytedance/seedance/v1.5/pro/text-to-video": {
        "class_name": "SeeDanceV15ProTextToVideo",
        "docstring": "SeeDance v1.5 Pro from ByteDance generates high-quality dance videos from text prompts.",
        "tags": ["video", "generation", "dance", "seedance", "bytedance", "text-to-video"],
        "use_cases": [
            "Generate dance choreography videos",
            "Create dance performance visualizations",
            "Produce music video concepts",
            "Generate dance training content",
            "Create dance animation prototypes"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/bytedance/seedance/v1/pro/fast/text-to-video": {
        "class_name": "SeeDanceV1ProFastTextToVideo",
        "docstring": "SeeDance v1 Pro Fast generates dance videos quickly from text with reduced generation time.",
        "tags": ["video", "generation", "dance", "seedance", "fast", "bytedance", "text-to-video"],
        "use_cases": [
            "Rapidly prototype dance videos",
            "Create quick dance previews",
            "Generate dance concepts efficiently",
            "Iterate on choreography ideas",
            "Produce dance storyboards"
        ],
        "basic_fields": ["prompt"]
    },
    
    "veed/fabric-1.0/text": {
        "class_name": "VeedFabric10Text",
        "docstring": "VEED Fabric 1.0 generates video content from text using advanced video synthesis.",
        "tags": ["video", "generation", "fabric", "veed", "text-to-video", "txt2vid"],
        "use_cases": [
            "Generate marketing videos from text",
            "Create explainer video content",
            "Produce video ads from copy",
            "Generate social media videos",
            "Create branded video content"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/ltx-video": {
        "class_name": "LTXVideo",
        "docstring": "LTX Video generates high-quality videos from text prompts with advanced temporal consistency.",
        "tags": ["video", "generation", "ltx", "text-to-video", "txt2vid"],
        "use_cases": [
            "Generate temporally consistent videos",
            "Create smooth video sequences",
            "Produce high-quality video content",
            "Generate professional video clips",
            "Create cinematic video scenes"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/kling-video/v1/standard/text-to-video": {
        "class_name": "KlingVideoV1StandardTextToVideo",
        "docstring": "Kling Video v1 Standard generates videos from text with balanced quality and speed.",
        "tags": ["video", "generation", "kling", "text-to-video", "txt2vid"],
        "use_cases": [
            "Generate standard quality videos",
            "Create video content efficiently",
            "Produce videos for web use",
            "Generate video previews",
            "Create video concepts"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/kling-video/v1/pro/text-to-video": {
        "class_name": "KlingVideoV1ProTextToVideo",
        "docstring": "Kling Video v1 Pro generates high-quality professional videos from text prompts.",
        "tags": ["video", "generation", "kling", "pro", "text-to-video", "txt2vid"],
        "use_cases": [
            "Generate professional grade videos",
            "Create high-quality marketing content",
            "Produce cinematic video sequences",
            "Generate detailed video scenes",
            "Create premium video content"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/mochi-v1": {
        "class_name": "MochiV1",
        "docstring": "Mochi v1 generates creative videos from text with unique artistic style.",
        "tags": ["video", "generation", "mochi", "artistic", "text-to-video", "txt2vid"],
        "use_cases": [
            "Generate artistic video content",
            "Create stylized animations",
            "Produce creative video art",
            "Generate experimental videos",
            "Create unique visual content"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/runway-gen3/turbo/text-to-video": {
        "class_name": "RunwayGen3TurboTextToVideo",
        "docstring": "Runway Gen-3 Turbo generates videos quickly from text with high quality output.",
        "tags": ["video", "generation", "runway", "gen3", "turbo", "text-to-video", "txt2vid"],
        "use_cases": [
            "Generate videos rapidly from text",
            "Create quick video prototypes",
            "Produce fast video iterations",
            "Generate real-time video content",
            "Create efficient video workflows"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/stable-video": {
        "class_name": "StableVideo",
        "docstring": "Stable Video generates consistent and stable video sequences from text prompts.",
        "tags": ["video", "generation", "stable", "text-to-video", "txt2vid"],
        "use_cases": [
            "Generate stable video sequences",
            "Create consistent video content",
            "Produce reliable video outputs",
            "Generate predictable video scenes",
            "Create controlled video generation"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/t2v-turbo": {
        "class_name": "T2VTurbo",
        "docstring": "T2V Turbo generates videos from text at high speed with optimized performance.",
        "tags": ["video", "generation", "turbo", "fast", "text-to-video", "txt2vid"],
        "use_cases": [
            "Generate videos with minimal latency",
            "Create rapid video prototypes",
            "Produce quick video previews",
            "Generate real-time video content",
            "Create efficient video workflows"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/wan-cinematic": {
        "class_name": "WanCinematic",
        "docstring": "Wan Cinematic generates cinematic quality videos from text with professional aesthetics.",
        "tags": ["video", "generation", "cinematic", "professional", "text-to-video", "txt2vid"],
        "use_cases": [
            "Generate cinematic video sequences",
            "Create film-quality content",
            "Produce professional video clips",
            "Generate movie-like scenes",
            "Create dramatic video content"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/minimax-video/v1": {
        "class_name": "MinimaxVideoV1",
        "docstring": "Minimax Video v1 generates videos from text with efficient resource usage.",
        "tags": ["video", "generation", "minimax", "text-to-video", "txt2vid"],
        "use_cases": [
            "Generate videos efficiently",
            "Create video content with minimal resources",
            "Produce lightweight video outputs",
            "Generate scalable video content",
            "Create optimized video workflows"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/minimax-video/v1/turbo": {
        "class_name": "MinimaxVideoV1Turbo",
        "docstring": "Minimax Video v1 Turbo generates videos from text at maximum speed.",
        "tags": ["video", "generation", "minimax", "turbo", "fast", "text-to-video", "txt2vid"],
        "use_cases": [
            "Generate videos at maximum speed",
            "Create rapid video iterations",
            "Produce instant video previews",
            "Generate real-time video responses",
            "Create ultra-fast video workflows"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/pyramidflow": {
        "class_name": "PyramidFlow",
        "docstring": "PyramidFlow generates videos with hierarchical processing for smooth motion.",
        "tags": ["video", "generation", "pyramid", "flow", "text-to-video", "txt2vid"],
        "use_cases": [
            "Generate smooth motion videos",
            "Create fluid video animations",
            "Produce high-quality motion sequences",
            "Generate temporally coherent videos",
            "Create professional motion graphics"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/luma-dream-machine": {
        "class_name": "LumaDreamMachineTextToVideo",
        "docstring": "Luma Dream Machine generates creative videos from text with dreamlike aesthetics.",
        "tags": ["video", "generation", "luma", "dream-machine", "text-to-video", "txt2vid"],
        "use_cases": [
            "Generate dreamlike video content",
            "Create surreal video sequences",
            "Produce artistic video interpretations",
            "Generate creative video concepts",
            "Create imaginative video art"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/luma-photon": {
        "class_name": "LumaPhoton",
        "docstring": "Luma Photon generates photorealistic videos from text with high visual fidelity.",
        "tags": ["video", "generation", "luma", "photon", "photorealistic", "text-to-video"],
        "use_cases": [
            "Generate photorealistic video content",
            "Create realistic video simulations",
            "Produce lifelike video scenes",
            "Generate high-fidelity video outputs",
            "Create realistic visual content"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/luma-photon-flash": {
        "class_name": "LumaPhotonFlashVideo",
        "docstring": "Luma Photon Flash generates photorealistic videos quickly with optimized speed.",
        "tags": ["video", "generation", "luma", "photon", "flash", "fast", "text-to-video"],
        "use_cases": [
            "Generate photorealistic videos rapidly",
            "Create realistic video previews",
            "Produce fast photorealistic content",
            "Generate quick realistic sequences",
            "Create efficient realistic workflows"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/luma-ray2": {
        "class_name": "LumaRay2",
        "docstring": "Luma Ray2 generates advanced video content with improved ray tracing techniques.",
        "tags": ["video", "generation", "luma", "ray2", "advanced", "text-to-video"],
        "use_cases": [
            "Generate ray-traced video content",
            "Create advanced lighting effects",
            "Produce high-quality rendered videos",
            "Generate realistic lighting sequences",
            "Create professional visual effects"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/luma-ray2/turbo": {
        "class_name": "LumaRay2Turbo",
        "docstring": "Luma Ray2 Turbo generates ray-traced videos with optimized rendering speed.",
        "tags": ["video", "generation", "luma", "ray2", "turbo", "fast", "text-to-video"],
        "use_cases": [
            "Generate ray-traced videos quickly",
            "Create fast rendered previews",
            "Produce efficient visual effects",
            "Generate rapid lighting iterations",
            "Create optimized rendering workflows"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/qihoo-t2v": {
        "class_name": "QihooT2V",
        "docstring": "Qihoo T2V generates videos from text with Chinese language optimization.",
        "tags": ["video", "generation", "qihoo", "chinese", "text-to-video", "txt2vid"],
        "use_cases": [
            "Generate videos from Chinese text",
            "Create multilingual video content",
            "Produce localized video scenes",
            "Generate culturally relevant videos",
            "Create international video content"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/wan-show-1": {
        "class_name": "WanShow1",
        "docstring": "Wan Show 1 generates presentation-style videos from text for showcasing ideas.",
        "tags": ["video", "generation", "presentation", "showcase", "text-to-video"],
        "use_cases": [
            "Generate presentation videos",
            "Create showcase content",
            "Produce pitch videos",
            "Generate demo videos",
            "Create educational presentations"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/luma-photon/v2": {
        "class_name": "LumaPhotonV2",
        "docstring": "Luma Photon v2 generates photorealistic videos with improved quality and detail.",
        "tags": ["video", "generation", "luma", "photon", "v2", "photorealistic", "text-to-video"],
        "use_cases": [
            "Generate high-quality photorealistic videos",
            "Create detailed realistic scenes",
            "Produce cinematic realistic content",
            "Generate professional video outputs",
            "Create premium photorealistic sequences"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/luma-dream-machine/v1.6": {
        "class_name": "LumaDreamMachineV16",
        "docstring": "Luma Dream Machine v1.6 generates creative videos with enhanced dream-like effects.",
        "tags": ["video", "generation", "luma", "dream-machine", "v1.6", "text-to-video"],
        "use_cases": [
            "Generate enhanced dreamlike videos",
            "Create surreal video art",
            "Produce creative visual content",
            "Generate artistic video sequences",
            "Create imaginative video effects"
        ],
        "basic_fields": ["prompt"]
    },

    "fal-ai/pixverse/v5.6/text-to-video": {
        "class_name": "PixverseV56TextToVideo",
        "docstring": "Pixverse",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/ltx-2-19b/distilled/text-to-video/lora": {
        "class_name": "Ltx219BDistilledTextToVideoLora",
        "docstring": "LTX-2 19B Distilled",
        "tags": ["video", "generation", "text-to-video", "txt2vid", "lora"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/ltx-2-19b/distilled/text-to-video": {
        "class_name": "Ltx219BDistilledTextToVideo",
        "docstring": "LTX-2 19B Distilled",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/ltx-2-19b/text-to-video/lora": {
        "class_name": "Ltx219BTextToVideoLora",
        "docstring": "LTX-2 19B",
        "tags": ["video", "generation", "text-to-video", "txt2vid", "lora"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/ltx-2-19b/text-to-video": {
        "class_name": "Ltx219BTextToVideo",
        "docstring": "LTX-2 19B",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/kandinsky5-pro/text-to-video": {
        "class_name": "Kandinsky5ProTextToVideo",
        "docstring": "Kandinsky5 Pro",
        "tags": ["video", "generation", "text-to-video", "txt2vid", "professional"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "wan/v2.6/text-to-video": {
        "class_name": "WanV26TextToVideo",
        "docstring": "Wan v2.6 Text to Video",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/kling-video/v2.6/pro/text-to-video": {
        "class_name": "KlingVideoV26ProTextToVideo",
        "docstring": "Kling Video v2.6 Text to Video",
        "tags": ["video", "generation", "text-to-video", "txt2vid", "professional"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/pixverse/v5.5/text-to-video": {
        "class_name": "PixverseV55TextToVideo",
        "docstring": "Pixverse",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/ltx-2/text-to-video/fast": {
        "class_name": "Ltx2TextToVideoFast",
        "docstring": "LTX Video 2.0 Fast",
        "tags": ["video", "generation", "text-to-video", "txt2vid", "fast"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/ltx-2/text-to-video": {
        "class_name": "Ltx2TextToVideo",
        "docstring": "LTX Video 2.0 Pro",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/hunyuan-video-v1.5/text-to-video": {
        "class_name": "HunyuanVideoV15TextToVideo",
        "docstring": "Hunyuan Video V1.5",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/infinity-star/text-to-video": {
        "class_name": "InfinityStarTextToVideo",
        "docstring": "Infinity Star",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/sana-video": {
        "class_name": "SanaVideo",
        "docstring": "Sana Video",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/longcat-video/text-to-video/720p": {
        "class_name": "LongcatVideoTextToVideo720P",
        "docstring": "LongCat Video",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/longcat-video/text-to-video/480p": {
        "class_name": "LongcatVideoTextToVideo480P",
        "docstring": "LongCat Video",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/longcat-video/distilled/text-to-video/720p": {
        "class_name": "LongcatVideoDistilledTextToVideo720P",
        "docstring": "LongCat Video Distilled",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/longcat-video/distilled/text-to-video/480p": {
        "class_name": "LongcatVideoDistilledTextToVideo480P",
        "docstring": "LongCat Video Distilled",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/minimax/hailuo-2.3/standard/text-to-video": {
        "class_name": "MinimaxHailuo23StandardTextToVideo",
        "docstring": "MiniMax Hailuo 2.3 [Standard] (Text to Video)",
        "tags": ["video", "generation", "text-to-video", "txt2vid", "professional"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/minimax/hailuo-2.3/pro/text-to-video": {
        "class_name": "MinimaxHailuo23ProTextToVideo",
        "docstring": "MiniMax Hailuo 2.3 [Pro] (Text to Video)",
        "tags": ["video", "generation", "text-to-video", "txt2vid", "professional"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/vidu/q2/text-to-video": {
        "class_name": "ViduQ2TextToVideo",
        "docstring": "Vidu",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/krea-wan-14b/text-to-video": {
        "class_name": "KreaWan14BTextToVideo",
        "docstring": "Krea Wan 14b- Text to Video",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/wan-alpha": {
        "class_name": "WanAlpha",
        "docstring": "Wan Alpha",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/kandinsky5/text-to-video/distill": {
        "class_name": "Kandinsky5TextToVideoDistill",
        "docstring": "Kandinsky5",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/kandinsky5/text-to-video": {
        "class_name": "Kandinsky5TextToVideo",
        "docstring": "Kandinsky5",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/veo3.1/fast": {
        "class_name": "Veo31Fast",
        "docstring": "Veo 3.1 Fast",
        "tags": ["video", "generation", "text-to-video", "txt2vid", "fast"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/veo3.1": {
        "class_name": "Veo31",
        "docstring": "Veo 3.1",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/sora-2/text-to-video/pro": {
        "class_name": "Sora2TextToVideoPro",
        "docstring": "Sora 2",
        "tags": ["video", "generation", "text-to-video", "txt2vid", "professional"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/sora-2/text-to-video": {
        "class_name": "Sora2TextToVideo",
        "docstring": "Sora 2",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/ovi": {
        "class_name": "Ovi",
        "docstring": "Ovi Text to Video",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/wan-25-preview/text-to-video": {
        "class_name": "Wan25PreviewTextToVideo",
        "docstring": "Wan 2.5 Text to Video",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/pixverse/v5/text-to-video": {
        "class_name": "PixverseV5TextToVideo",
        "docstring": "Pixverse",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/infinitalk/single-text": {
        "class_name": "InfinitalkSingleText",
        "docstring": "Infinitalk",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "moonvalley/marey/t2v": {
        "class_name": "MoonvalleyMareyT2V",
        "docstring": "Marey Realism V1.5",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production"

        ],
    },
    "fal-ai/wan/v2.2-a14b/text-to-video/lora": {
        "class_name": "WanV22A14bTextToVideoLora",
        "docstring": "Wan-2.2 text-to-video is a video model that generates high-quality videos with high visual quality and motion diversity from text prompts. This endpoint supports LoRAs made for Wan 2.2.",
        "tags": ["video", "generation", "text-to-video", "txt2vid", "lora"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/wan/v2.2-5b/text-to-video/distill": {
        "class_name": "WanV225bTextToVideoDistill",
        "docstring": "Wan 2.2's 5B distill model produces up to 5 seconds of video 720p at 24FPS with fluid motion and powerful prompt understanding",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/wan/v2.2-5b/text-to-video/fast-wan": {
        "class_name": "WanV225bTextToVideoFastWan",
        "docstring": "Wan 2.2's 5B FastVideo model produces up to 5 seconds of video 720p at 24FPS with fluid motion and powerful prompt understanding",
        "tags": ["video", "generation", "text-to-video", "txt2vid", "fast"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/wan/v2.2-a14b/text-to-video/turbo": {
        "class_name": "WanV22A14bTextToVideoTurbo",
        "docstring": "Wan-2.2 turbo text-to-video is a video model that generates high-quality videos with high visual quality and motion diversity from text prompts. ",
        "tags": ["video", "generation", "text-to-video", "txt2vid", "fast"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/wan/v2.2-5b/text-to-video": {
        "class_name": "WanV225bTextToVideo",
        "docstring": "Wan 2.2's 5B model produces up to 5 seconds of video 720p at 24FPS with fluid motion and powerful prompt understanding",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/wan/v2.2-a14b/text-to-video": {
        "class_name": "WanV22A14bTextToVideo",
        "docstring": "Wan-2.2 text-to-video is a video model that generates high-quality videos with high visual quality and motion diversity from text prompts. ",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/ltxv-13b-098-distilled": {
        "class_name": "Ltxv13b098Distilled",
        "docstring": "Generate long videos from prompts using LTX Video-0.9.8 13B Distilled and custom LoRA",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/minimax/hailuo-02/pro/text-to-video": {
        "class_name": "MinimaxHailuo02ProTextToVideo",
        "docstring": "MiniMax Hailuo-02 Text To Video API (Pro, 1080p): Advanced video generation model with 1080p resolution",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/bytedance/seedance/v1/pro/text-to-video": {
        "class_name": "BytedanceSeedanceV1ProTextToVideo",
        "docstring": "Seedance 1.0 Pro, a high quality video generation model developed by Bytedance.",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/bytedance/seedance/v1/lite/text-to-video": {
        "class_name": "BytedanceSeedanceV1LiteTextToVideo",
        "docstring": "Seedance 1.0 Lite",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/kling-video/v2.1/master/text-to-video": {
        "class_name": "KlingVideoV21MasterTextToVideo",
        "docstring": "Kling 2.1 Master: The premium endpoint for Kling 2.1, designed for top-tier text-to-video generation with unparalleled motion fluidity, cinematic visuals, and exceptional prompt precision.",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/ltx-video-13b-dev": {
        "class_name": "LtxVideo13bDev",
        "docstring": "Generate videos from prompts using LTX Video-0.9.7 13B and custom LoRA",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/ltx-video-13b-distilled": {
        "class_name": "LtxVideo13bDistilled",
        "docstring": "Generate videos from prompts using LTX Video-0.9.7 13B Distilled and custom LoRA",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/pixverse/v4.5/text-to-video/fast": {
        "class_name": "PixverseV45TextToVideoFast",
        "docstring": "Generate high quality and fast video clips from text and image prompts using PixVerse v4.5 fast",
        "tags": ["video", "generation", "text-to-video", "txt2vid", "fast"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/pixverse/v4.5/text-to-video": {
        "class_name": "PixverseV45TextToVideo",
        "docstring": "Generate high quality video clips from text and image prompts using PixVerse v4.5",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/vidu/q1/text-to-video": {
        "class_name": "ViduQ1TextToVideo",
        "docstring": "Vidu Q1 Text to Video generates high-quality 1080p videos with exceptional visual quality and motion diversity",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/magi": {
        "class_name": "Magi",
        "docstring": "MAGI-1 is a video generation model with exceptional understanding of physical interactions and cinematic prompts",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/magi-distilled": {
        "class_name": "MagiDistilled",
        "docstring": "MAGI-1 distilled is a faster video generation model with exceptional understanding of physical interactions and cinematic prompts",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/pixverse/v4/text-to-video": {
        "class_name": "PixverseV4TextToVideo",
        "docstring": "Generate high quality video clips from text and image prompts using PixVerse v4",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/pixverse/v4/text-to-video/fast": {
        "class_name": "PixverseV4TextToVideoFast",
        "docstring": "Generate high quality and fast video clips from text and image prompts using PixVerse v4 fast",
        "tags": ["video", "generation", "text-to-video", "txt2vid", "fast"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/kling-video/lipsync/audio-to-video": {
        "class_name": "KlingVideoLipsyncAudioToVideo",
        "docstring": "Kling LipSync is an audio-to-video model that generates realistic lip movements from audio input.",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/kling-video/lipsync/text-to-video": {
        "class_name": "KlingVideoLipsyncTextToVideo",
        "docstring": "Kling LipSync is a text-to-video model that generates realistic lip movements from text input.",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/wan-t2v-lora": {
        "class_name": "WanT2vLora",
        "docstring": "Add custom LoRAs to Wan-2.1 is a text-to-video model that generates high-quality videos with high visual quality and motion diversity from images",
        "tags": ["video", "generation", "text-to-video", "txt2vid", "lora"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/luma-dream-machine/ray-2-flash": {
        "class_name": "LumaDreamMachineRay2Flash",
        "docstring": "Ray2 Flash is a fast video generative model capable of creating realistic visuals with natural, coherent motion.",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/pika/v2.1/text-to-video": {
        "class_name": "PikaV21TextToVideo",
        "docstring": "Start with a simple text input to create dynamic generations that defy expectations. Anything you dream can come to life with sharp details, impressive character control and cinematic camera moves.",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/pika/v2.2/text-to-video": {
        "class_name": "PikaV22TextToVideo",
        "docstring": "Start with a simple text input to create dynamic generations that defy expectations in up to 1080p. Experience better image clarity and crisper, sharper visuals.",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/pika/v2/turbo/text-to-video": {
        "class_name": "PikaV2TurboTextToVideo",
        "docstring": "Pika v2 Turbo creates videos from a text prompt with high quality output.",
        "tags": ["video", "generation", "text-to-video", "txt2vid", "fast"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/wan-pro/text-to-video": {
        "class_name": "WanProTextToVideo",
        "docstring": "Wan-2.1 Pro is a premium text-to-video model that generates high-quality 1080p videos at 30fps with up to 6 seconds duration, delivering exceptional visual quality and motion diversity from text prompts",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/kling-video/v1.6/pro/effects": {
        "class_name": "KlingVideoV16ProEffects",
        "docstring": "Generate video clips from your prompts using Kling 1.6 (pro)",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/kling-video/v1.6/standard/effects": {
        "class_name": "KlingVideoV16StandardEffects",
        "docstring": "Generate video clips from your prompts using Kling 1.6 (std)",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/kling-video/v1.5/pro/effects": {
        "class_name": "KlingVideoV15ProEffects",
        "docstring": "Generate video clips from your prompts using Kling 1.5 (pro)",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/kling-video/v1/standard/effects": {
        "class_name": "KlingVideoV1StandardEffects",
        "docstring": "Generate video clips from your prompts using Kling 1.0",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/ltx-video-v095": {
        "class_name": "LtxVideoV095",
        "docstring": "Generate videos from prompts using LTX Video-0.9.5",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/kling-video/v1.6/pro/text-to-video": {
        "class_name": "KlingVideoV16ProTextToVideo",
        "docstring": "Generate video clips from your prompts using Kling 1.6 (pro)",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/wan/v2.1/1.3b/text-to-video": {
        "class_name": "WanV2113bTextToVideo",
        "docstring": "Wan-2.1 1.3B is a text-to-video model that generates high-quality videos with high visual quality and motion diversity from text promptsat faster speeds.",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/wan-t2v": {
        "class_name": "WanT2v",
        "docstring": "Wan-2.1 is a text-to-video model that generates high-quality videos with high visual quality and motion diversity from text prompts",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/veo2": {
        "class_name": "Veo2",
        "docstring": "Veo 2 creates videos with realistic motion and high quality output. Explore different styles and find your own with extensive camera controls.",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/minimax/video-01-director": {
        "class_name": "MinimaxVideo01Director",
        "docstring": "Generate video clips more accurately with respect to natural language descriptions and using camera movement instructions for shot control.",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/pixverse/v3.5/text-to-video": {
        "class_name": "PixverseV35TextToVideo",
        "docstring": "Generate high quality video clips from text prompts using PixVerse v3.5",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/pixverse/v3.5/text-to-video/fast": {
        "class_name": "PixverseV35TextToVideoFast",
        "docstring": "Generate high quality video clips quickly from text prompts using PixVerse v3.5 Fast",
        "tags": ["video", "generation", "text-to-video", "txt2vid", "fast"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/luma-dream-machine/ray-2": {
        "class_name": "LumaDreamMachineRay2",
        "docstring": "Ray2 is a large-scale video generative model capable of creating realistic visuals with natural, coherent motion.",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/hunyuan-video-lora": {
        "class_name": "HunyuanVideoLora",
        "docstring": "Hunyuan Video is an Open video generation model with high visual quality, motion diversity, text-video alignment, and generation stability",
        "tags": ["video", "generation", "text-to-video", "txt2vid", "lora"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/transpixar": {
        "class_name": "Transpixar",
        "docstring": "Transform text into stunning videos with TransPixar - an AI model that generates both RGB footage and alpha channels, enabling seamless compositing and creative video effects.",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/kling-video/v1.6/standard/text-to-video": {
        "class_name": "KlingVideoV16StandardTextToVideo",
        "docstring": "Generate video clips from your prompts using Kling 1.6 (std)",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/minimax/video-01-live": {
        "class_name": "MinimaxVideo01Live",
        "docstring": "Generate video clips from your prompts using MiniMax model",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/kling-video/v1.5/pro/text-to-video": {
        "class_name": "KlingVideoV15ProTextToVideo",
        "docstring": "Generate video clips from your prompts using Kling 1.5 (pro)",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/fast-svd/text-to-video": {
        "class_name": "FastSvdTextToVideo",
        "docstring": "Generate short video clips from your prompts using SVD v1.1",
        "tags": ["video", "generation", "text-to-video", "txt2vid", "fast"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/fast-svd-lcm/text-to-video": {
        "class_name": "FastSvdLcmTextToVideo",
        "docstring": "Generate short video clips from your images using SVD v1.1 at Lightning Speed",
        "tags": ["video", "generation", "text-to-video", "txt2vid", "fast"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
    "fal-ai/minimax/video-01": {
        "class_name": "MinimaxVideo01",
        "docstring": "Generate video clips from your prompts using MiniMax model",
        "tags": ["video", "generation", "text-to-video", "txt2vid"],
        "use_cases": [
            "AI-generated video content",
            "Marketing and advertising videos",
            "Educational content creation",
            "Social media video posts",
            "Automated video production",
        ],
    },
}


def get_config(endpoint_id: str) -> dict[str, Any]:
    """Get config for an endpoint."""
    return CONFIGS.get(endpoint_id, {})
