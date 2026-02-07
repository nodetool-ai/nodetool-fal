import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nodetool.nodes.fal.text_to_video import (
    Veo3,
    Veo3AspectRatio,
    Veo3Duration,
    Veo3Resolution,
)
from nodetool.nodes.fal.fal_node import FALNode


class TestTextToVideoNodeImports:
    """Test that text-to-video nodes can be imported correctly."""

    def test_import_veo3(self):
        """Test that Veo3 node can be imported."""
        assert Veo3 is not None
        assert issubclass(Veo3, FALNode)


class TestTextToVideoEnums:
    """Test enum definitions for text-to-video nodes."""

    def test_veo3_aspect_ratio_values(self):
        """Test Veo3AspectRatio contains expected values."""
        assert Veo3AspectRatio.RATIO_16_9.value == "16:9"
        assert Veo3AspectRatio.RATIO_9_16.value == "9:16"

    def test_veo3_duration_values(self):
        """Test Veo3Duration contains expected values."""
        assert Veo3Duration.FOUR_SECONDS.value == "4s"
        assert Veo3Duration.SIX_SECONDS.value == "6s"
        assert Veo3Duration.EIGHT_SECONDS.value == "8s"

    def test_veo3_resolution_values(self):
        """Test Veo3Resolution contains expected values."""
        assert Veo3Resolution.RES_720P.value == "720p"
        assert Veo3Resolution.RES_1080P.value == "1080p"


class TestTextToVideoNodeVisibility:
    """Test node visibility settings for text-to-video."""

    def test_veo3_is_visible(self):
        """Veo3 node should be visible."""
        assert Veo3.is_visible() is True


class TestTextToVideoNodeInstantiation:
    """Test that text-to-video nodes can be instantiated with default values."""

    def test_veo3_instantiation(self):
        """Test Veo3 node instantiation."""
        node = Veo3()
        assert node.prompt == ""
        assert node.aspect_ratio == Veo3AspectRatio.RATIO_16_9
        assert node.duration == Veo3Duration.EIGHT_SECONDS
        assert node.resolution == Veo3Resolution.RES_720P
        assert node.generate_audio is True
        assert node.seed == -1
        assert node.auto_fix is True


class TestTextToVideoBasicFields:
    """Test get_basic_fields method on text-to-video nodes."""

    def test_veo3_basic_fields(self):
        """Test Veo3 basic fields."""
        fields = Veo3.get_basic_fields()
        assert isinstance(fields, list)
        assert len(fields) > 0
