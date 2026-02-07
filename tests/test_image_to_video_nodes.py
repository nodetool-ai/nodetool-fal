import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nodetool.nodes.fal.image_to_video import (
    LumaDreamMachine,
    KlingVideo,
    PixverseV56ImageToVideo,
    AspectRatio,
    KlingDuration,
    PixverseV56AspectRatio,
    PixverseV56Resolution,
    PixverseV56Duration,
    PixverseV56Style,
)
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.metadata.types import ImageRef


class TestImageToVideoNodeImports:
    """Test that image-to-video nodes can be imported correctly."""

    def test_import_luma_dream_machine(self):
        """Test that LumaDreamMachine node can be imported."""
        assert LumaDreamMachine is not None
        assert issubclass(LumaDreamMachine, FALNode)

    def test_import_kling_video(self):
        """Test that KlingVideo node can be imported."""
        assert KlingVideo is not None
        assert issubclass(KlingVideo, FALNode)

    def test_import_pixverse_v56(self):
        """Test that PixverseV56ImageToVideo node can be imported."""
        assert PixverseV56ImageToVideo is not None
        assert issubclass(PixverseV56ImageToVideo, FALNode)


class TestImageToVideoEnums:
    """Test enum definitions for image-to-video nodes."""

    def test_aspect_ratio_values(self):
        """Test AspectRatio contains expected values."""
        assert AspectRatio.RATIO_16_9.value == "16:9"
        assert AspectRatio.RATIO_9_16.value == "9:16"
        assert AspectRatio.RATIO_4_3.value == "4:3"

    def test_pixverse_aspect_ratio_values(self):
        """Test PixverseV56AspectRatio contains expected values."""
        assert PixverseV56AspectRatio.RATIO_16_9.value == "16:9"
        assert PixverseV56AspectRatio.RATIO_9_16.value == "9:16"


class TestImageToVideoNodeVisibility:
    """Test node visibility settings for image-to-video."""

    def test_luma_dream_machine_is_visible(self):
        """LumaDreamMachine node should be visible."""
        assert LumaDreamMachine.is_visible() is True

    def test_kling_video_is_visible(self):
        """KlingVideo node should be visible."""
        assert KlingVideo.is_visible() is True

    def test_pixverse_v56_is_visible(self):
        """PixverseV56ImageToVideo node should be visible."""
        assert PixverseV56ImageToVideo.is_visible() is True


class TestImageToVideoNodeInstantiation:
    """Test that image-to-video nodes can be instantiated with default values."""

    def test_luma_dream_machine_instantiation(self):
        """Test LumaDreamMachine node instantiation."""
        node = LumaDreamMachine()
        assert isinstance(node.image, ImageRef)
        assert node.prompt == ""
        assert node.aspect_ratio == AspectRatio.RATIO_16_9
        assert node.loop is False

    def test_kling_video_instantiation(self):
        """Test KlingVideo node instantiation."""
        node = KlingVideo()
        assert isinstance(node.image, ImageRef)
        assert node.prompt == ""

    def test_pixverse_v56_instantiation(self):
        """Test PixverseV56ImageToVideo node instantiation."""
        node = PixverseV56ImageToVideo()
        assert isinstance(node.image, ImageRef)
        assert node.prompt == ""


class TestImageToVideoBasicFields:
    """Test get_basic_fields method on image-to-video nodes."""

    def test_luma_dream_machine_basic_fields(self):
        """Test LumaDreamMachine basic fields."""
        if hasattr(LumaDreamMachine, "get_basic_fields"):
            fields = LumaDreamMachine.get_basic_fields()
            assert isinstance(fields, list)
            assert len(fields) > 0

    def test_kling_video_basic_fields(self):
        """Test KlingVideo basic fields."""
        if hasattr(KlingVideo, "get_basic_fields"):
            fields = KlingVideo.get_basic_fields()
            assert isinstance(fields, list)
            assert len(fields) > 0

    def test_pixverse_v56_basic_fields(self):
        """Test PixverseV56ImageToVideo basic fields."""
        if hasattr(PixverseV56ImageToVideo, "get_basic_fields"):
            fields = PixverseV56ImageToVideo.get_basic_fields()
            assert isinstance(fields, list)
            assert len(fields) > 0
