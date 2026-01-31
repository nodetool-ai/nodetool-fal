import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nodetool.nodes.fal.fal_node import FALNode
from nodetool.nodes.fal.llm import AnyLLM, ModelEnum
from nodetool.nodes.fal.text_to_image import (
    IdeogramV2,
    ImageSizePreset,
    StylePreset,
    AspectRatio,
    IdeogramStyle,
)


class TestNodeImports:
    """Test that all node modules can be imported correctly."""

    def test_import_llm_nodes(self):
        """Test that LLM nodes can be imported."""
        assert AnyLLM is not None
        assert issubclass(AnyLLM, FALNode)

    def test_import_text_to_image_nodes(self):
        """Test that text-to-image nodes can be imported."""
        assert IdeogramV2 is not None
        assert issubclass(IdeogramV2, FALNode)


class TestEnums:
    """Test enum definitions."""

    def test_model_enum_values(self):
        """Test ModelEnum contains expected models."""
        assert ModelEnum.CLAUDE_3_SONNET.value == "anthropic/claude-3.5-sonnet"
        assert ModelEnum.GEMINI_FLASH.value == "google/gemini-flash-1.5"
        assert ModelEnum.GPT4.value == "openai/gpt-4o"
        assert ModelEnum.LLAMA_70B.value == "meta-llama/llama-3.1-70b-instruct"

    def test_image_size_preset_values(self):
        """Test ImageSizePreset contains expected values."""
        assert ImageSizePreset.SQUARE_HD.value == "square_hd"
        assert ImageSizePreset.PORTRAIT_16_9.value == "portrait_16_9"
        assert ImageSizePreset.LANDSCAPE_4_3.value == "landscape_4_3"

    def test_style_preset_values(self):
        """Test StylePreset contains expected values."""
        assert StylePreset.REALISTIC_IMAGE.value == "realistic_image"
        assert StylePreset.ANIME.value == "anime"
        assert StylePreset.WATERCOLOR.value == "watercolor"
        assert StylePreset._3D_RENDER.value == "3d_render"

    def test_aspect_ratio_values(self):
        """Test AspectRatio contains expected values."""
        assert AspectRatio.RATIO_1_1.value == "1:1"
        assert AspectRatio.RATIO_16_9.value == "16:9"
        assert AspectRatio.RATIO_9_16.value == "9:16"
        assert AspectRatio.RATIO_4_3.value == "4:3"

    def test_ideogram_style_values(self):
        """Test IdeogramStyle contains expected values."""
        assert IdeogramStyle.AUTO.value == "auto"
        assert IdeogramStyle.REALISTIC.value == "realistic"
        assert IdeogramStyle.ANIME.value == "anime"
        assert IdeogramStyle.RENDER_3D.value == "render_3D"


class TestNodeVisibility:
    """Test node visibility settings."""

    def test_anyLLM_is_visible(self):
        """AnyLLM node should be visible."""
        assert AnyLLM.is_visible() is True

    def test_ideogramV2_is_visible(self):
        """IdeogramV2 node should be visible."""
        assert IdeogramV2.is_visible() is True


class TestNodeBasicFields:
    """Test get_basic_fields method on nodes."""

    def test_anyLLM_basic_fields(self):
        """Test AnyLLM basic fields."""
        fields = AnyLLM.get_basic_fields()
        assert "prompt" in fields
        assert "model" in fields

    def test_ideogramV2_basic_fields(self):
        """Test IdeogramV2 basic fields."""
        # IdeogramV2 should have get_basic_fields if implemented
        if hasattr(IdeogramV2, "get_basic_fields"):
            fields = IdeogramV2.get_basic_fields()
            assert isinstance(fields, list)
