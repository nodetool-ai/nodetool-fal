"""
Type definitions for FAL nodes.

This module contains BaseType subclass definitions used by generated FAL nodes.
These types represent complex nested structures in the FAL API.
"""

from nodetool.metadata.types import BaseType
from pydantic import Field
from typing import Any


class AudioTimeSpan(BaseType):
    """Audio time span specification."""
    type: str = "audio_time_span"
    start: float = Field(default=0.0, description="Start time in seconds")
    end: float = Field(default=0.0, description="End time in seconds")


class BBoxPromptBase(BaseType):
    """Bounding box prompt base."""
    type: str = "bbox_prompt_base"
    x: int = Field(default=0, description="X coordinate")
    y: int = Field(default=0, description="Y coordinate")
    width: int = Field(default=0, description="Width")
    height: int = Field(default=0, description="Height")


class BoxPrompt(BaseType):
    """Box prompt for image segmentation."""
    type: str = "box_prompt"
    x: int = Field(default=0, description="X coordinate")
    y: int = Field(default=0, description="Y coordinate")
    width: int = Field(default=0, description="Width")
    height: int = Field(default=0, description="Height")


class BoxPromptBase(BaseType):
    """Base class for box prompts."""
    type: str = "box_prompt_base"
    x: int = Field(default=0, description="X coordinate")
    y: int = Field(default=0, description="Y coordinate")
    width: int = Field(default=0, description="Width")
    height: int = Field(default=0, description="Height")


class ChronoLoraWeight(BaseType):
    """Chrono LoRA weight configuration."""
    type: str = "chrono_lora_weight"
    path: str = Field(default="", description="Path to LoRA weights")
    scale: float = Field(default=1.0, description="Weight scale")


class ControlLoraWeight(BaseType):
    """Control LoRA weight configuration."""
    type: str = "control_lora_weight"
    path: str = Field(default="", description="Path to LoRA weights")
    scale: float = Field(default=1.0, description="Weight scale")


class ControlNet(BaseType):
    """ControlNet configuration."""
    type: str = "control_net"
    path: str = Field(default="", description="ControlNet model path")
    conditioning_scale: float = Field(default=1.0, description="Conditioning scale")


class ControlNetUnion(BaseType):
    """ControlNet Union configuration."""
    type: str = "control_net_union"
    path: str = Field(default="", description="ControlNet model path")
    conditioning_scale: float = Field(default=1.0, description="Conditioning scale")


class DialogueBlock(BaseType):
    """Dialogue block for text-to-speech."""
    type: str = "dialogue_block"
    speaker: str = Field(default="", description="Speaker identifier")
    text: str = Field(default="", description="Text content")


class DynamicMask(BaseType):
    """Dynamic mask specification."""
    type: str = "dynamic_mask"
    mask_data: Any = Field(default=None, description="Mask data")


class EasyControlWeight(BaseType):
    """Easy control weight configuration."""
    type: str = "easy_control_weight"
    scale: float = Field(default=1.0, description="Weight scale")


class ElementInput(BaseType):
    """Element input specification."""
    type: str = "element_input"
    element_data: Any = Field(default=None, description="Element data")


class Embedding(BaseType):
    """Embedding configuration."""
    type: str = "embedding"
    path: str = Field(default="", description="Embedding path")
    tokens: list[str] = Field(default_factory=list, description="Token list")


class Frame(BaseType):
    """Frame specification for video."""
    type: str = "frame"
    index: int = Field(default=0, description="Frame index")
    data: Any = Field(default=None, description="Frame data")


class GuidanceInput(BaseType):
    """Guidance input for image generation."""
    type: str = "guidance_input"
    guidance_data: Any = Field(default=None, description="Guidance data")


class IPAdapter(BaseType):
    """IP-Adapter configuration."""
    type: str = "ip_adapter"
    image_encoder_path: str = Field(default="", description="Image encoder path")
    ip_adapter_path: str = Field(default="", description="IP-Adapter model path")
    scale: float = Field(default=1.0, description="Adapter scale")


class ImageCondition(BaseType):
    """Image conditioning specification."""
    type: str = "image_condition"
    image_url: str = Field(default="", description="Image URL")
    conditioning_scale: float = Field(default=1.0, description="Conditioning scale")


class ImageConditioningInput(BaseType):
    """Image conditioning input."""
    type: str = "image_conditioning_input"
    image_url: str = Field(default="", description="Image URL")
    conditioning_scale: float = Field(default=1.0, description="Conditioning scale")


class ImageInput(BaseType):
    """Image input specification."""
    type: str = "image_input"
    image_url: str = Field(default="", description="Image URL")


class InpaintSection(BaseType):
    """Inpainting section specification."""
    type: str = "inpaint_section"
    start: float = Field(default=0.0, description="Start position")
    end: float = Field(default=0.0, description="End position")


class KeyframeTransition(BaseType):
    """Keyframe transition specification."""
    type: str = "keyframe_transition"
    frame: int = Field(default=0, description="Frame number")
    transition_type: str = Field(default="linear", description="Transition type")


class KlingV3ComboElementInput(BaseType):
    """Kling V3 combo element input."""
    type: str = "kling_v3_combo_element_input"
    element_data: Any = Field(default=None, description="Element data")


class KlingV3MultiPromptElement(BaseType):
    """Kling V3 multi-prompt element."""
    type: str = "kling_v3_multi_prompt_element"
    prompt: str = Field(default="", description="Prompt text")
    weight: float = Field(default=1.0, description="Prompt weight")


class LoRAInput(BaseType):
    """LoRA input configuration."""
    type: str = "lora_input"
    path: str = Field(default="", description="LoRA model path")
    scale: float = Field(default=1.0, description="LoRA scale")


class LoRAWeight(BaseType):
    """LoRA weight configuration."""
    type: str = "lora_weight"
    path: str = Field(default="", description="LoRA weights path")
    scale: float = Field(default=1.0, description="Weight scale")


class LoraWeight(BaseType):
    """LoRA weight configuration (alternate naming)."""
    type: str = "lora_weight"
    path: str = Field(default="", description="LoRA weights path")
    scale: float = Field(default=1.0, description="Weight scale")


class MoondreamInputParam(BaseType):
    """Moondream input parameters."""
    type: str = "moondream_input_param"
    param_data: Any = Field(default=None, description="Parameter data")


class OmniVideoElementInput(BaseType):
    """Omni video element input."""
    type: str = "omni_video_element_input"
    element_data: Any = Field(default=None, description="Element data")


class PointPrompt(BaseType):
    """Point prompt for image segmentation."""
    type: str = "point_prompt"
    x: int = Field(default=0, description="X coordinate")
    y: int = Field(default=0, description="Y coordinate")
    label: int = Field(default=1, description="Point label")


class PointPromptBase(BaseType):
    """Base class for point prompts."""
    type: str = "point_prompt_base"
    x: int = Field(default=0, description="X coordinate")
    y: int = Field(default=0, description="Y coordinate")


class PronunciationDictionaryLocator(BaseType):
    """Pronunciation dictionary locator."""
    type: str = "pronunciation_dictionary_locator"
    dictionary_id: str = Field(default="", description="Dictionary ID")


class RGBColor(BaseType):
    """RGB color specification."""
    type: str = "rgb_color"
    r: int = Field(default=0, ge=0, le=255, description="Red component")
    g: int = Field(default=0, ge=0, le=255, description="Green component")
    b: int = Field(default=0, ge=0, le=255, description="Blue component")


class ReferenceFace(BaseType):
    """Reference face for face-related operations."""
    type: str = "reference_face"
    image_url: str = Field(default="", description="Face image URL")


class ReferenceImageInput(BaseType):
    """Reference image input."""
    type: str = "reference_image_input"
    image_url: str = Field(default="", description="Reference image URL")


class SemanticImageInput(BaseType):
    """Semantic image input."""
    type: str = "semantic_image_input"
    image_url: str = Field(default="", description="Semantic image URL")


class Speaker(BaseType):
    """Speaker configuration."""
    type: str = "speaker"
    speaker_id: str = Field(default="", description="Speaker identifier")
    voice: str = Field(default="", description="Voice identifier")


class Track(BaseType):
    """Audio/video track specification."""
    type: str = "track"
    track_id: str = Field(default="", description="Track identifier")
    track_data: Any = Field(default=None, description="Track data")


class Turn(BaseType):
    """Conversation turn specification."""
    type: str = "turn"
    role: str = Field(default="user", description="Turn role (user/assistant)")
    content: str = Field(default="", description="Turn content")


class VibeVoiceSpeaker(BaseType):
    """Vibe voice speaker configuration."""
    type: str = "vibe_voice_speaker"
    speaker_id: str = Field(default="", description="Speaker identifier")
    voice_style: str = Field(default="", description="Voice style")


class VideoCondition(BaseType):
    """Video conditioning specification."""
    type: str = "video_condition"
    video_url: str = Field(default="", description="Video URL")
    conditioning_scale: float = Field(default=1.0, description="Conditioning scale")


class VideoConditioningInput(BaseType):
    """Video conditioning input."""
    type: str = "video_conditioning_input"
    video_url: str = Field(default="", description="Video URL")
    conditioning_scale: float = Field(default=1.0, description="Conditioning scale")
