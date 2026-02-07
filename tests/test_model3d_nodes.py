import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nodetool.nodes.fal.model3d import (
    Trellis,
    Hunyuan3DV2,
    TripoSR,
    Era3D,
    TextureSizeEnum,
)
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.metadata.types import ImageRef, Model3DRef


class TestModel3DNodeImports:
    """Test that all 3D model nodes can be imported correctly."""

    def test_import_trellis(self):
        """Test that Trellis node can be imported."""
        assert Trellis is not None
        assert issubclass(Trellis, FALNode)

    def test_import_hunyuan3d(self):
        """Test that Hunyuan3DV2 node can be imported."""
        assert Hunyuan3DV2 is not None
        assert issubclass(Hunyuan3DV2, FALNode)

    def test_import_triposr(self):
        """Test that TripoSR node can be imported."""
        assert TripoSR is not None
        assert issubclass(TripoSR, FALNode)

    def test_import_era3d(self):
        """Test that Era3D node can be imported."""
        assert Era3D is not None
        assert issubclass(Era3D, FALNode)


class TestModel3DEnums:
    """Test enum definitions for 3D model nodes."""

    def test_texture_size_enum(self):
        """Test TextureSizeEnum contains expected values."""
        assert TextureSizeEnum.SIZE_512.value == 512
        assert TextureSizeEnum.SIZE_1024.value == 1024
        assert TextureSizeEnum.SIZE_2048.value == 2048


class TestModel3DNodeVisibility:
    """Test node visibility settings for 3D models."""

    def test_trellis_is_visible(self):
        """Trellis node should be visible."""
        assert Trellis.is_visible() is True

    def test_hunyuan3d_is_visible(self):
        """Hunyuan3DV2 node should be visible."""
        assert Hunyuan3DV2.is_visible() is True

    def test_triposr_is_visible(self):
        """TripoSR node should be visible."""
        assert TripoSR.is_visible() is True

    def test_era3d_is_visible(self):
        """Era3D node should be visible."""
        assert Era3D.is_visible() is True


class TestModel3DBasicFields:
    """Test get_basic_fields method on 3D model nodes."""

    def test_trellis_basic_fields(self):
        """Test Trellis basic fields."""
        fields = Trellis.get_basic_fields()
        assert "image" in fields
        assert "texture_size" in fields

    def test_hunyuan3d_basic_fields(self):
        """Test Hunyuan3DV2 basic fields."""
        fields = Hunyuan3DV2.get_basic_fields()
        assert "image" in fields

    def test_triposr_basic_fields(self):
        """Test TripoSR basic fields."""
        fields = TripoSR.get_basic_fields()
        assert "image" in fields

    def test_era3d_basic_fields(self):
        """Test Era3D basic fields."""
        fields = Era3D.get_basic_fields()
        assert "image" in fields


class TestModel3DNodeInstantiation:
    """Test that 3D model nodes can be instantiated with default values."""

    def test_trellis_instantiation(self):
        """Test Trellis node instantiation."""
        node = Trellis()
        assert isinstance(node.image, ImageRef)
        assert node.ss_guidance_strength == 7.5
        assert node.ss_sampling_steps == 12
        assert node.texture_size == TextureSizeEnum.SIZE_1024
        assert node.seed == -1

    def test_hunyuan3d_instantiation(self):
        """Test Hunyuan3DV2 node instantiation."""
        node = Hunyuan3DV2()
        assert isinstance(node.image, ImageRef)
        assert node.num_inference_steps == 50
        assert node.guidance_scale == 2.0
        assert node.seed == -1

    def test_triposr_instantiation(self):
        """Test TripoSR node instantiation."""
        node = TripoSR()
        assert isinstance(node.image, ImageRef)
        assert node.foreground_ratio == 0.85

    def test_era3d_instantiation(self):
        """Test Era3D node instantiation."""
        node = Era3D()
        assert isinstance(node.image, ImageRef)
        assert node.num_inference_steps == 40
        assert node.seed == -1


class TestModel3DReturnTypes:
    """Test return types for 3D model nodes."""

    def test_triposr_return_type(self):
        """Test TripoSR return type is Model3DRef."""
        return_type = TripoSR.return_type()
        assert return_type == Model3DRef

    def test_era3d_output_type(self):
        """Test Era3D has OutputType definition."""
        assert hasattr(Era3D, "OutputType")
        output_type = Era3D.OutputType
        assert "mv_images" in output_type.__annotations__
        assert "model" in output_type.__annotations__
