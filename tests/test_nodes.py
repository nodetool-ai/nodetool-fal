import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nodetool.nodes.fal.fal_node import FALNode
from nodetool.nodes.fal.llm import OpenRouter
from nodetool.nodes.fal.text_to_image import (
    IdeogramV2,
    ImageSizePreset,
)


class TestNodeImports:
    """Test that all node modules can be imported correctly."""

    def test_import_llm_nodes(self):
        """Test that LLM nodes can be imported."""
        assert OpenRouter is not None
        assert issubclass(OpenRouter, FALNode)

    def test_import_text_to_image_nodes(self):
        """Test that text-to-image nodes can be imported."""
        assert IdeogramV2 is not None
        assert issubclass(IdeogramV2, FALNode)


class TestEnums:
    """Test enum definitions."""

    def test_image_size_preset_values(self):
        """Test ImageSizePreset contains expected values."""
        assert ImageSizePreset.SQUARE_HD.value == "square_hd"
        assert ImageSizePreset.PORTRAIT_16_9.value == "portrait_16_9"
        assert ImageSizePreset.LANDSCAPE_4_3.value == "landscape_4_3"

    def test_ideogram_aspect_ratio_values(self):
        """Test IdeogramV2 nested AspectRatio enum values."""
        assert IdeogramV2.AspectRatio.RATIO_1_1.value == "1:1"
        assert IdeogramV2.AspectRatio.RATIO_16_9.value == "16:9"
        assert IdeogramV2.AspectRatio.RATIO_9_16.value == "9:16"
        assert IdeogramV2.AspectRatio.RATIO_4_3.value == "4:3"

    def test_ideogram_style_values(self):
        """Test IdeogramV2 nested Style enum values."""
        assert IdeogramV2.Style.AUTO.value == "auto"
        assert IdeogramV2.Style.REALISTIC.value == "realistic"
        assert IdeogramV2.Style.ANIME.value == "anime"
        assert IdeogramV2.Style.RENDER_3D.value == "render_3D"


class TestNodeVisibility:
    """Test node visibility settings."""

    def test_openrouter_is_visible(self):
        """OpenRouter node should be visible."""
        assert OpenRouter.is_visible() is True

    def test_ideogram_v2_is_visible(self):
        """IdeogramV2 node should be visible."""
        assert IdeogramV2.is_visible() is True


class TestNodeBasicFields:
    """Test get_basic_fields method on nodes."""

    def test_openrouter_basic_fields(self):
        """Test OpenRouter basic fields."""
        fields = OpenRouter.get_basic_fields()
        assert "prompt" in fields
        assert "model" in fields

    def test_ideogram_v2_basic_fields(self):
        """Test IdeogramV2 basic fields."""
        if hasattr(IdeogramV2, "get_basic_fields"):
            fields = IdeogramV2.get_basic_fields()
            assert isinstance(fields, list)


class TestNewModuleImports:
    """Test that all new module categories can be imported."""

    def test_import_audio_to_text(self):
        from nodetool.nodes.fal import audio_to_text
        assert audio_to_text is not None

    def test_import_image_to_json(self):
        from nodetool.nodes.fal import image_to_json
        assert image_to_json is not None

    def test_import_json_processing(self):
        from nodetool.nodes.fal import json_processing
        assert json_processing is not None

    def test_import_speech_to_speech(self):
        from nodetool.nodes.fal import speech_to_speech
        assert speech_to_speech is not None

    def test_import_text_to_3d(self):
        from nodetool.nodes.fal import text_to_3d
        assert text_to_3d is not None

    def test_import_text_to_json(self):
        from nodetool.nodes.fal import text_to_json
        assert text_to_json is not None

    def test_import_text_to_text(self):
        from nodetool.nodes.fal import text_to_text
        assert text_to_text is not None

    def test_import_unknown(self):
        from nodetool.nodes.fal import unknown
        assert unknown is not None

    def test_import_video_to_audio(self):
        from nodetool.nodes.fal import video_to_audio
        assert video_to_audio is not None

    def test_import_video_to_text(self):
        from nodetool.nodes.fal import video_to_text
        assert video_to_text is not None
