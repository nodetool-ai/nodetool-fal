"""
Shared BaseType subclasses for FAL node parameters.

These types correspond to OpenAPI $ref schemas that appear as array items
in multiple FAL endpoints (e.g. LoraWeight, ControlLoraWeight).
"""

from typing import Literal

from pydantic import Field

from nodetool.metadata.types import BaseType


# ---------------------------------------------------------------------------
# LoRA types
# ---------------------------------------------------------------------------


class LoraWeight(BaseType):
    """A LoRA weight configuration for image/video generation."""

    type: Literal["fal_lora_weight"] = "fal_lora_weight"
    path: str = Field(default="", description="URL or the path to the LoRA weights.")
    scale: float = Field(
        default=1.0,
        description="The scale of the LoRA weight.",
    )


class LoRAWeight(BaseType):
    """A LoRA weight configuration (WAN / LTX style with weight_name)."""

    type: Literal["fal_lora_weight_v2"] = "fal_lora_weight_v2"
    path: str = Field(default="", description="URL or the path to the LoRA weights.")
    scale: float = Field(default=1.0, description="The scale of the LoRA weight.")
    transformer: str = Field(default="high", description="Transformer setting.")
    weight_name: str = Field(default="", description="Weight name.")


class LoRAInput(BaseType):
    """A LoRA input for z-image and similar endpoints."""

    type: Literal["fal_lora_input"] = "fal_lora_input"
    path: str = Field(default="", description="URL or the path to the LoRA weights.")
    scale: float = Field(default=1.0, description="The scale of the LoRA weight.")


class ControlLoraWeight(BaseType):
    """A ControlNet LoRA weight configuration with a control image."""

    type: Literal["fal_control_lora_weight"] = "fal_control_lora_weight"
    path: str = Field(default="", description="URL or the path to the LoRA weights.")
    scale: float = Field(
        default=1.0,
        description="The scale of the LoRA weight.",
    )
    control_image_url: str = Field(
        default="",
        description="URL of the image to be used as the control image.",
    )
    preprocess: str = Field(
        default="None",
        description="Type of preprocessing to apply to the input image.",
    )


class ChronoLoraWeight(BaseType):
    """A LoRA weight configuration for ChronoEdit models."""

    type: Literal["fal_chrono_lora_weight"] = "fal_chrono_lora_weight"
    path: str = Field(
        default="", description="URL or path to the LoRA weights (Safetensors)."
    )
    scale: float = Field(
        default=1.0, description="Scale factor controlling LoRA strength."
    )


class EasyControlWeight(BaseType):
    """An EasyControl weight configuration."""

    type: Literal["fal_easy_control_weight"] = "fal_easy_control_weight"
    scale: float = Field(default=1.0, description="The scale of the weight.")
    image_control_type: str = Field(
        default="", description="Type of image control."
    )
    control_method_url: str = Field(
        default="", description="URL of the control method."
    )
    image_url: str = Field(default="", description="URL of the image.")


# ---------------------------------------------------------------------------
# ControlNet / IP-Adapter / Embedding types
# ---------------------------------------------------------------------------


class ControlNet(BaseType):
    """A ControlNet configuration."""

    type: Literal["fal_controlnet"] = "fal_controlnet"
    path: str = Field(default="", description="Path to the ControlNet model.")
    control_image_url: str = Field(
        default="", description="URL of the control image."
    )
    conditioning_scale: float = Field(
        default=1.0, description="Conditioning scale."
    )
    start_percentage: float = Field(default=0.0, description="Start percentage.")
    end_percentage: float = Field(default=1.0, description="End percentage.")
    mask_image_url: str = Field(default="", description="URL of the mask image.")
    mask_threshold: float = Field(default=0.5, description="Mask threshold.")
    variant: str = Field(default="", description="Model variant.")
    config_url: str = Field(default="", description="Config URL.")


class ControlNetUnion(BaseType):
    """A ControlNet Union configuration."""

    type: Literal["fal_controlnet_union"] = "fal_controlnet_union"
    path: str = Field(default="", description="Path to the ControlNet Union model.")
    controls: list[dict] = Field(default=[], description="Control inputs.")
    variant: str = Field(default="", description="Model variant.")
    config_url: str = Field(default="", description="Config URL.")


class ControlNetUnionInput(BaseType):
    """A single input for ControlNet Union."""

    type: Literal["fal_controlnet_union_input"] = "fal_controlnet_union_input"
    control_image_url: str = Field(
        default="", description="URL of the control image."
    )
    control_mode: str = Field(default="", description="Control mode.")
    conditioning_scale: float = Field(
        default=1.0, description="Conditioning scale."
    )
    start_percentage: float = Field(default=0.0, description="Start percentage.")
    end_percentage: float = Field(default=1.0, description="End percentage.")
    mask_image_url: str = Field(default="", description="URL of the mask image.")
    mask_threshold: float = Field(default=0.5, description="Mask threshold.")


class IPAdapter(BaseType):
    """An IP-Adapter configuration."""

    type: Literal["fal_ip_adapter"] = "fal_ip_adapter"
    path: str = Field(default="", description="Path to the IP-Adapter model.")
    image_url: str = Field(default="", description="URL of the input image.")
    scale: float = Field(default=1.0, description="Scale factor.")
    image_encoder_path: str = Field(
        default="", description="Path to the image encoder."
    )
    mask_image_url: str = Field(default="", description="URL of the mask image.")
    mask_threshold: float = Field(default=0.5, description="Mask threshold.")
    image_encoder_weight_name: str = Field(
        default="", description="Image encoder weight name."
    )
    image_encoder_subfolder: str = Field(
        default="", description="Image encoder subfolder."
    )
    subfolder: str = Field(default="", description="Subfolder.")
    weight_name: str = Field(default="", description="Weight name.")


class Embedding(BaseType):
    """A text embedding configuration."""

    type: Literal["fal_embedding"] = "fal_embedding"
    path: str = Field(default="", description="Path to the embedding.")
    tokens: list[str] = Field(
        default=["<s0>", "<s1>"], description="Embedding tokens."
    )


# ---------------------------------------------------------------------------
# Color types
# ---------------------------------------------------------------------------


class RGBColor(BaseType):
    """An RGB color value."""

    type: Literal["fal_rgb_color"] = "fal_rgb_color"
    r: int = Field(default=0, description="Red channel (0-255).")
    g: int = Field(default=0, description="Green channel (0-255).")
    b: int = Field(default=0, description="Blue channel (0-255).")


# ---------------------------------------------------------------------------
# Prompt / Segmentation types
# ---------------------------------------------------------------------------


class PointPrompt(BaseType):
    """A point prompt for segmentation models (SAM2)."""

    type: Literal["fal_point_prompt"] = "fal_point_prompt"
    x: int = Field(default=0, description="X coordinate.")
    y: int = Field(default=0, description="Y coordinate.")
    label: int = Field(default=1, description="Label (1 for foreground, 0 for background).")
    frame_index: int = Field(default=0, description="Frame index.")


class PointPromptBase(BaseType):
    """A point prompt for SAM3 models."""

    type: Literal["fal_point_prompt_base"] = "fal_point_prompt_base"
    x: int = Field(default=0, description="X coordinate.")
    y: int = Field(default=0, description="Y coordinate.")
    label: int = Field(default=0, description="Label.")
    object_id: int = Field(default=0, description="Object ID.")


class BoxPrompt(BaseType):
    """A box prompt for segmentation models (SAM2)."""

    type: Literal["fal_box_prompt"] = "fal_box_prompt"
    x_min: int = Field(default=0, description="Minimum X coordinate.")
    y_min: int = Field(default=0, description="Minimum Y coordinate.")
    x_max: int = Field(default=0, description="Maximum X coordinate.")
    y_max: int = Field(default=0, description="Maximum Y coordinate.")
    frame_index: int = Field(default=0, description="Frame index.")


class BoxPromptBase(BaseType):
    """A box prompt for SAM3 and similar models."""

    type: Literal["fal_box_prompt_base"] = "fal_box_prompt_base"
    x_min: int = Field(default=0, description="Minimum X coordinate.")
    y_min: int = Field(default=0, description="Minimum Y coordinate.")
    x_max: int = Field(default=0, description="Maximum X coordinate.")
    y_max: int = Field(default=0, description="Maximum Y coordinate.")
    object_id: int = Field(default=0, description="Object ID.")


class BBoxPromptBase(BaseType):
    """A bounding box prompt for object removal."""

    type: Literal["fal_bbox_prompt_base"] = "fal_bbox_prompt_base"
    x_min: float = Field(default=0.0, description="Minimum X coordinate.")
    y_min: float = Field(default=0.0, description="Minimum Y coordinate.")
    x_max: float = Field(default=0.0, description="Maximum X coordinate.")
    y_max: float = Field(default=0.0, description="Maximum Y coordinate.")


# ---------------------------------------------------------------------------
# Kling types
# ---------------------------------------------------------------------------


class KlingV3MultiPromptElement(BaseType):
    """A single shot element for Kling V3 multi-shot video generation."""

    type: Literal["kling_v3_multi_prompt_element"] = "kling_v3_multi_prompt_element"
    prompt: str = Field(default="", description="The prompt for this shot.")
    duration: str = Field(
        default="5", description="The duration of this shot in seconds (3-15)."
    )


class KlingV3ComboElementInput(BaseType):
    """An element (character/object) for Kling V3 video generation."""

    type: Literal["kling_v3_combo_element_input"] = "kling_v3_combo_element_input"
    frontal_image_url: str = Field(
        default="", description="The frontal image URL of the element (main view)."
    )
    reference_image_urls: list[str] = Field(
        default=[],
        description="Additional reference image URLs from different angles.",
    )
    video_url: str = Field(
        default="",
        description="The video URL of the element.",
    )


class ElementInput(BaseType):
    """An element input for Kling image generation."""

    type: Literal["fal_element_input"] = "fal_element_input"
    frontal_image_url: str = Field(
        default="", description="The frontal image URL."
    )
    reference_image_urls: list[str] = Field(
        default=[], description="Additional reference image URLs."
    )


class OmniVideoElementInput(BaseType):
    """An element input for OmniVideo reference-to-video generation."""

    type: Literal["fal_omni_video_element_input"] = "fal_omni_video_element_input"
    frontal_image_url: str = Field(
        default="", description="The frontal image URL."
    )
    reference_image_urls: list[str] = Field(
        default=[], description="Additional reference image URLs."
    )


class DynamicMask(BaseType):
    """A dynamic mask for Kling video generation."""

    type: Literal["fal_dynamic_mask"] = "fal_dynamic_mask"
    mask_url: str = Field(default="", description="URL of the mask image.")
    trajectories: list[dict] = Field(
        default=[], description="Trajectory points (x, y)."
    )


# ---------------------------------------------------------------------------
# Video / Frame types
# ---------------------------------------------------------------------------


class Frame(BaseType):
    """A video frame reference."""

    type: Literal["fal_frame"] = "fal_frame"
    url: str = Field(default="", description="URL of the frame image.")


class KeyframeTransition(BaseType):
    """A keyframe transition for Pika models."""

    type: Literal["fal_keyframe_transition"] = "fal_keyframe_transition"
    prompt: str = Field(default="", description="Prompt for the transition.")
    duration: int = Field(default=5, description="Duration in seconds.")


class ImageCondition(BaseType):
    """An image condition for LTX multiconditioning."""

    type: Literal["fal_image_condition"] = "fal_image_condition"
    image_url: str = Field(default="", description="URL of the image.")
    strength: float = Field(default=1.0, description="Conditioning strength.")
    start_frame_number: int = Field(default=0, description="Start frame number.")


class VideoCondition(BaseType):
    """A video condition for LTX multiconditioning."""

    type: Literal["fal_video_condition"] = "fal_video_condition"
    video_url: str = Field(default="", description="URL of the video.")
    strength: float = Field(default=1.0, description="Conditioning strength.")
    start_frame_number: int = Field(default=0, description="Start frame number.")


class ImageConditioningInput(BaseType):
    """An image conditioning input for LTX distilled multiconditioning."""

    type: Literal["fal_image_conditioning_input"] = "fal_image_conditioning_input"
    image_url: str = Field(default="", description="URL of the image.")
    strength: float = Field(default=1.0, description="Conditioning strength.")
    start_frame_num: int = Field(default=0, description="Start frame number.")


class VideoConditioningInput(BaseType):
    """A video conditioning input for LTX distilled multiconditioning."""

    type: Literal["fal_video_conditioning_input"] = "fal_video_conditioning_input"
    video_url: str = Field(default="", description="URL of the video.")
    strength: float = Field(default=1.0, description="Conditioning strength.")
    start_frame_num: int = Field(default=0, description="Start frame number.")
    reverse_video: bool = Field(default=False, description="Reverse the video.")
    limit_num_frames: bool = Field(default=False, description="Limit number of frames.")
    resample_fps: bool = Field(default=False, description="Resample FPS.")
    target_fps: int = Field(default=24, description="Target FPS.")
    max_num_frames: int = Field(default=1441, description="Maximum number of frames.")
    conditioning_type: str = Field(default="rgb", description="Conditioning type.")
    preprocess: bool = Field(default=False, description="Whether to preprocess.")


class Track(BaseType):
    """A media track for FFmpeg compose."""

    type: Literal["fal_track"] = "fal_track"
    track_type: str = Field(default="", description="Type of the track.")
    id: str = Field(default="", description="Track ID.")
    keyframes: list[dict] = Field(default=[], description="Keyframes.")


# ---------------------------------------------------------------------------
# Audio types
# ---------------------------------------------------------------------------


class AudioTimeSpan(BaseType):
    """A time span for audio processing."""

    type: Literal["fal_audio_time_span"] = "fal_audio_time_span"
    start: float = Field(default=0.0, description="Start time in seconds.")
    end: float = Field(default=0.0, description="End time in seconds.")
    include: bool = Field(default=True, description="Whether to include this span.")


class InpaintSection(BaseType):
    """An inpaint section for audio generation."""

    type: Literal["fal_inpaint_section"] = "fal_inpaint_section"
    start: float = Field(default=0.0, description="Start time in seconds.")
    end: float = Field(default=0.0, description="End time in seconds.")


class DialogueBlock(BaseType):
    """A dialogue block for text-to-dialogue generation."""

    type: Literal["fal_dialogue_block"] = "fal_dialogue_block"
    text: str = Field(default="", description="The dialogue text.")
    voice: str = Field(default="", description="The voice to use.")


class PronunciationDictionaryLocator(BaseType):
    """A pronunciation dictionary locator for ElevenLabs."""

    type: Literal["fal_pronunciation_dict_locator"] = "fal_pronunciation_dict_locator"
    pronunciation_dictionary_id: str = Field(
        default="", description="Pronunciation dictionary ID."
    )
    version_id: str = Field(default="", description="Version ID.")


class Speaker(BaseType):
    """A speaker configuration for CSM models."""

    type: Literal["fal_speaker"] = "fal_speaker"
    speaker_id: int = Field(default=0, description="Speaker ID.")
    prompt: str = Field(default="", description="Speaker prompt text.")
    audio_url: str = Field(default="", description="URL of the speaker audio.")


class Turn(BaseType):
    """A conversation turn for CSM models."""

    type: Literal["fal_turn"] = "fal_turn"
    text: str = Field(default="", description="The text to speak.")
    speaker_id: int = Field(default=0, description="Speaker ID.")


class VibeVoiceSpeaker(BaseType):
    """A speaker configuration for VibeVoice models."""

    type: Literal["fal_vibe_voice_speaker"] = "fal_vibe_voice_speaker"
    preset: str = Field(default="Alice [EN]", description="Voice preset.")
    audio_url: str = Field(default="", description="URL of the speaker audio.")


# ---------------------------------------------------------------------------
# Vision / Guidance types
# ---------------------------------------------------------------------------


class GuidanceInput(BaseType):
    """A guidance input for Bria text-to-image."""

    type: Literal["fal_guidance_input"] = "fal_guidance_input"
    image_url: str = Field(default="", description="URL of the guidance image.")
    scale: float = Field(default=1.0, description="Guidance scale.")
    method: str = Field(default="", description="Guidance method.")


class ReferenceFace(BaseType):
    """A reference face for PuLID models."""

    type: Literal["fal_reference_face"] = "fal_reference_face"
    image_url: str = Field(default="", description="URL of the reference face image.")


class MoondreamInputParam(BaseType):
    """An input parameter for Moondream batched inference."""

    type: Literal["fal_moondream_input_param"] = "fal_moondream_input_param"
    prompt: str = Field(
        default="Describe this image.", description="The prompt for inference."
    )
    image_url: str = Field(default="", description="URL of the input image.")


class ImageInput(BaseType):
    """An image input for Arbiter measurements."""

    type: Literal["fal_image_input"] = "fal_image_input"
    hypothesis: str = Field(default="", description="Hypothesis text.")


class ReferenceImageInput(BaseType):
    """A reference image input for Arbiter measurements."""

    type: Literal["fal_reference_image_input"] = "fal_reference_image_input"
    hypothesis: str = Field(default="", description="Hypothesis text.")
    reference: str = Field(default="", description="Reference text.")


class SemanticImageInput(BaseType):
    """A semantic image input for Arbiter measurements."""

    type: Literal["fal_semantic_image_input"] = "fal_semantic_image_input"
    hypothesis: str = Field(default="", description="Hypothesis text.")
    reference: str = Field(default="", description="Reference text.")
