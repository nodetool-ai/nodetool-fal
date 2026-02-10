from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.metadata.types import ImageRef
from nodetool.nodes.fal.types import Track
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class OpenRouter(FALNode):
    """
    OpenRouter provides unified access to any LLM (Large Language Model) through a single API.
    llm, chat, openrouter, multimodel, language-model

    Use cases:
    - Run any LLM through unified interface
    - Switch between models seamlessly
    - Access multiple LLM providers
    - Flexible model selection
    - Unified LLM API access
    """

    prompt: str = Field(
        default="", description="Prompt to be used for the chat completion"
    )
    model: str = Field(
        default="", description="Name of the model to use. Charged based on actual token usage."
    )
    max_tokens: int = Field(
        default=0, description="This sets the upper limit for the number of tokens the model can generate in response. It won't produce more than this limit. The maximum value is the context length minus the prompt length."
    )
    temperature: float = Field(
        default=1, description="This setting influences the variety in the model's responses. Lower values lead to more predictable and typical responses, while higher values encourage more diverse and less common responses. At 0, the model always gives the same response for a given input."
    )
    system_prompt: str = Field(
        default="", description="System prompt to provide context or instructions to the model"
    )
    reasoning: bool = Field(
        default=False, description="Should reasoning be the part of the final answer."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "prompt": self.prompt,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system_prompt": self.system_prompt,
            "reasoning": self.reasoning,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="openrouter/router",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "model"]

class OpenRouterChatCompletions(FALNode):
    """
    OpenRouter Chat Completions provides OpenAI-compatible interface for any LLM.
    llm, chat, openai-compatible, openrouter, chat-completions

    Use cases:
    - OpenAI-compatible LLM access
    - Drop-in replacement for OpenAI API
    - Multi-model chat completions
    - Standardized chat interface
    - Universal LLM chat API
    """


    async def process(self, context: ProcessingContext) -> Any:
        arguments = {
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="openrouter/router/openai/v1/chat/completions",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["messages", "model"]

class Qwen3Guard(FALNode):
    """
    Qwen 3 Guard provides content safety and moderation using Qwen's LLM.
    llm, safety, moderation, qwen, guard

    Use cases:
    - Content safety checking
    - Moderation of text content
    - Safety filtering for outputs
    - Content policy enforcement
    - Text safety analysis
    """

    class Label(Enum):
        """
        The classification label
        """
        SAFE = "Safe"
        UNSAFE = "Unsafe"
        CONTROVERSIAL = "Controversial"


    prompt: str = Field(
        default="", description="The input text to be classified"
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "prompt": self.prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-3-guard",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class OpenrouterRouterOpenaiV1Responses(FALNode):
    """
    The OpenRouter Responses API with fal, powered by OpenRouter, provides unified access to a wide range of large language models - including GPT, Claude, Gemini, and many others through a single API interface.
    llm, language-model, text-generation, ai

    Use cases:
    - Text generation and completion
    - Conversational AI
    - Content summarization
    - Code generation
    - Creative writing assistance
    """


    async def process(self, context: ProcessingContext) -> Any:
        arguments = {
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="openrouter/router/openai/v1/responses",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class OpenrouterRouterOpenaiV1Embeddings(FALNode):
    """
    The OpenRouter Embeddings API with fal, powered by OpenRouter, provides unified access to a wide range of large language models - including GPT, Claude, Gemini, and many others through a single API interface.
    llm, language-model, text-generation, ai

    Use cases:
    - Text generation and completion
    - Conversational AI
    - Content summarization
    - Code generation
    - Creative writing assistance
    """


    async def process(self, context: ProcessingContext) -> Any:
        arguments = {
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="openrouter/router/openai/v1/embeddings",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["text"]

class VideoPromptGenerator(FALNode):
    """
    Generate video prompts using a variety of techniques including camera direction, style, pacing, special effects and more.
    llm, language-model, text-generation, ai

    Use cases:
    - Text generation and completion
    - Conversational AI
    - Content summarization
    - Code generation
    - Creative writing assistance
    """

    class Style(Enum):
        """
        Style of the video prompt
        """
        MINIMALIST = "Minimalist"
        SIMPLE = "Simple"
        DETAILED = "Detailed"
        DESCRIPTIVE = "Descriptive"
        DYNAMIC = "Dynamic"
        CINEMATIC = "Cinematic"
        DOCUMENTARY = "Documentary"
        ANIMATION = "Animation"
        ACTION = "Action"
        EXPERIMENTAL = "Experimental"

    class CameraDirection(Enum):
        """
        Camera direction
        """
        NONE = "None"
        ZOOM_IN = "Zoom in"
        ZOOM_OUT = "Zoom out"
        PAN_LEFT = "Pan left"
        PAN_RIGHT = "Pan right"
        TILT_UP = "Tilt up"
        TILT_DOWN = "Tilt down"
        ORBITAL_ROTATION = "Orbital rotation"
        PUSH_IN = "Push in"
        PULL_OUT = "Pull out"
        TRACK_FORWARD = "Track forward"
        TRACK_BACKWARD = "Track backward"
        SPIRAL_IN = "Spiral in"
        SPIRAL_OUT = "Spiral out"
        ARC_MOVEMENT = "Arc movement"
        DIAGONAL_TRAVERSE = "Diagonal traverse"
        VERTICAL_RISE = "Vertical rise"
        VERTICAL_DESCENT = "Vertical descent"

    class Pacing(Enum):
        """
        Pacing rhythm
        """
        NONE = "None"
        SLOW_BURN = "Slow burn"
        RHYTHMIC_PULSE = "Rhythmic pulse"
        FRANTIC_ENERGY = "Frantic energy"
        EBB_AND_FLOW = "Ebb and flow"
        HYPNOTIC_DRIFT = "Hypnotic drift"
        TIME_LAPSE_RUSH = "Time-lapse rush"
        STOP_MOTION_STACCATO = "Stop-motion staccato"
        GRADUAL_BUILD = "Gradual build"
        QUICK_CUT_RHYTHM = "Quick cut rhythm"
        LONG_TAKE_MEDITATION = "Long take meditation"
        JUMP_CUT_ENERGY = "Jump cut energy"
        MATCH_CUT_FLOW = "Match cut flow"
        CROSS_DISSOLVE_DREAMSCAPE = "Cross-dissolve dreamscape"
        PARALLEL_ACTION = "Parallel action"
        SLOW_MOTION_IMPACT = "Slow motion impact"
        RAMPING_DYNAMICS = "Ramping dynamics"
        MONTAGE_TEMPO = "Montage tempo"
        CONTINUOUS_FLOW = "Continuous flow"
        EPISODIC_BREAKS = "Episodic breaks"

    class SpecialEffects(Enum):
        """
        Special effects approach
        """
        NONE = "None"
        PRACTICAL_EFFECTS = "Practical effects"
        CGI_ENHANCEMENT = "CGI enhancement"
        ANALOG_GLITCHES = "Analog glitches"
        LIGHT_PAINTING = "Light painting"
        PROJECTION_MAPPING = "Projection mapping"
        NANOSECOND_EXPOSURES = "Nanosecond exposures"
        DOUBLE_EXPOSURE = "Double exposure"
        SMOKE_DIFFUSION = "Smoke diffusion"
        LENS_FLARE_ARTISTRY = "Lens flare artistry"
        PARTICLE_SYSTEMS = "Particle systems"
        HOLOGRAPHIC_OVERLAY = "Holographic overlay"
        CHROMATIC_ABERRATION = "Chromatic aberration"
        DIGITAL_DISTORTION = "Digital distortion"
        WIRE_REMOVAL = "Wire removal"
        MOTION_CAPTURE = "Motion capture"
        MINIATURE_INTEGRATION = "Miniature integration"
        WEATHER_SIMULATION = "Weather simulation"
        COLOR_GRADING = "Color grading"
        MIXED_MEDIA_COMPOSITE = "Mixed media composite"
        NEURAL_STYLE_TRANSFER = "Neural style transfer"

    class Model(Enum):
        """
        Model to use
        """
        ANTHROPIC_CLAUDE_3_5_SONNET = "anthropic/claude-3.5-sonnet"
        ANTHROPIC_CLAUDE_3_5_HAIKU = "anthropic/claude-3-5-haiku"
        ANTHROPIC_CLAUDE_3_HAIKU = "anthropic/claude-3-haiku"
        GOOGLE_GEMINI_2_5_FLASH_LITE = "google/gemini-2.5-flash-lite"
        GOOGLE_GEMINI_2_0_FLASH_001 = "google/gemini-2.0-flash-001"
        META_LLAMA_LLAMA_3_2_1B_INSTRUCT = "meta-llama/llama-3.2-1b-instruct"
        META_LLAMA_LLAMA_3_2_3B_INSTRUCT = "meta-llama/llama-3.2-3b-instruct"
        META_LLAMA_LLAMA_3_1_8B_INSTRUCT = "meta-llama/llama-3.1-8b-instruct"
        META_LLAMA_LLAMA_3_1_70B_INSTRUCT = "meta-llama/llama-3.1-70b-instruct"
        OPENAI_GPT_4O_MINI = "openai/gpt-4o-mini"
        OPENAI_GPT_4O = "openai/gpt-4o"
        DEEPSEEK_DEEPSEEK_R1 = "deepseek/deepseek-r1"

    class CameraStyle(Enum):
        """
        Camera movement style
        """
        NONE = "None"
        STEADICAM_FLOW = "Steadicam flow"
        DRONE_AERIALS = "Drone aerials"
        HANDHELD_URGENCY = "Handheld urgency"
        CRANE_ELEGANCE = "Crane elegance"
        DOLLY_PRECISION = "Dolly precision"
        VR_360 = "VR 360"
        MULTI_ANGLE_RIG = "Multi-angle rig"
        STATIC_TRIPOD = "Static tripod"
        GIMBAL_SMOOTHNESS = "Gimbal smoothness"
        SLIDER_MOTION = "Slider motion"
        JIB_SWEEP = "Jib sweep"
        POV_IMMERSION = "POV immersion"
        TIME_SLICE_ARRAY = "Time-slice array"
        MACRO_EXTREME = "Macro extreme"
        TILT_SHIFT_MINIATURE = "Tilt-shift miniature"
        SNORRICAM_CHARACTER = "Snorricam character"
        WHIP_PAN_DYNAMICS = "Whip pan dynamics"
        DUTCH_ANGLE_TENSION = "Dutch angle tension"
        UNDERWATER_HOUSING = "Underwater housing"
        PERISCOPE_LENS = "Periscope lens"

    class PromptLength(Enum):
        """
        Length of the prompt
        """
        SHORT = "Short"
        MEDIUM = "Medium"
        LONG = "Long"


    custom_elements: str = Field(
        default="", description="Custom technical elements (optional)"
    )
    style: Style = Field(
        default=Style.SIMPLE, description="Style of the video prompt"
    )
    camera_direction: CameraDirection = Field(
        default=CameraDirection.NONE, description="Camera direction"
    )
    pacing: Pacing = Field(
        default=Pacing.NONE, description="Pacing rhythm"
    )
    special_effects: SpecialEffects = Field(
        default=SpecialEffects.NONE, description="Special effects approach"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of an image to analyze and incorporate into the video prompt (optional)"
    )
    model: Model = Field(
        default=Model.GOOGLE_GEMINI_2_0_FLASH_001, description="Model to use"
    )
    camera_style: CameraStyle = Field(
        default=CameraStyle.NONE, description="Camera movement style"
    )
    input_concept: str = Field(
        default="", description="Core concept or thematic input for the video prompt"
    )
    prompt_length: PromptLength = Field(
        default=PromptLength.MEDIUM, description="Length of the prompt"
    )

    async def process(self, context: ProcessingContext) -> Any:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "custom_elements": self.custom_elements,
            "style": self.style.value,
            "camera_direction": self.camera_direction.value,
            "pacing": self.pacing.value,
            "special_effects": self.special_effects.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "model": self.model.value,
            "camera_style": self.camera_style.value,
            "input_concept": self.input_concept,
            "prompt_length": self.prompt_length.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/video-prompt-generator",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["text"]