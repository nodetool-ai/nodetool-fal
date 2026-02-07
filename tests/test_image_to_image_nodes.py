import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nodetool.nodes.fal.image_to_image import (
    FluxProRedux,
    FluxDevRedux,
    FluxLoraDepth,
)
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.metadata.types import ImageRef


class TestImageToImageNodeImports:
    """Test that image-to-image nodes can be imported correctly."""

    def test_import_flux_dev_redux(self):
        """Test that FluxDevRedux node can be imported."""
        assert FluxDevRedux is not None
        assert issubclass(FluxDevRedux, FALNode)

    def test_import_flux_pro_redux(self):
        """Test that FluxProRedux node can be imported."""
        assert FluxProRedux is not None
        assert issubclass(FluxProRedux, FALNode)

    def test_import_flux_lora_depth(self):
        """Test that FluxLoraDepth node can be imported."""
        assert FluxLoraDepth is not None
        assert issubclass(FluxLoraDepth, FALNode)


class TestImageToImageNodeVisibility:
    """Test node visibility settings for image-to-image."""

    def test_flux_dev_redux_is_visible(self):
        """FluxDevRedux node should be visible."""
        assert FluxDevRedux.is_visible() is True

    def test_flux_pro_redux_is_visible(self):
        """FluxProRedux node should be visible."""
        assert FluxProRedux.is_visible() is True

    def test_flux_lora_depth_is_visible(self):
        """FluxLoraDepth node should be visible."""
        assert FluxLoraDepth.is_visible() is True


class TestImageToImageNodeInstantiation:
    """Test that image-to-image nodes can be instantiated with default values."""

    def test_flux_dev_redux_instantiation(self):
        """Test FluxDevRedux node instantiation."""
        node = FluxDevRedux()
        assert isinstance(node.image, ImageRef)

    def test_flux_pro_redux_instantiation(self):
        """Test FluxProRedux node instantiation."""
        node = FluxProRedux()
        assert isinstance(node.image, ImageRef)

    def test_flux_lora_depth_instantiation(self):
        """Test FluxLoraDepth node instantiation."""
        node = FluxLoraDepth()
        assert node.prompt == ""
        assert isinstance(node.image, ImageRef)


class TestImageToImageBasicFields:
    """Test get_basic_fields method on image-to-image nodes."""

    def test_flux_dev_redux_basic_fields(self):
        """Test FluxDevRedux basic fields."""
        if hasattr(FluxDevRedux, "get_basic_fields"):
            fields = FluxDevRedux.get_basic_fields()
            assert isinstance(fields, list)
            assert len(fields) > 0

    def test_flux_pro_redux_basic_fields(self):
        """Test FluxProRedux basic fields."""
        if hasattr(FluxProRedux, "get_basic_fields"):
            fields = FluxProRedux.get_basic_fields()
            assert isinstance(fields, list)
            assert len(fields) > 0

    def test_flux_lora_depth_basic_fields(self):
        """Test FluxLoraDepth basic fields."""
        if hasattr(FluxLoraDepth, "get_basic_fields"):
            fields = FluxLoraDepth.get_basic_fields()
            assert isinstance(fields, list)
            assert len(fields) > 0
