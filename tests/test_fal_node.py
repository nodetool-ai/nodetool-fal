import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nodetool.nodes.fal.fal_node import FALNode


class TestFALNode:
    """Tests for the FALNode base class."""

    def test_is_visible_base_class(self):
        """FALNode base class should not be visible."""
        assert FALNode.is_visible() is False

    def test_is_visible_subclass(self):
        """Subclasses of FALNode should be visible."""

        class MyFALNode(FALNode):
            pass

        assert MyFALNode.is_visible() is True

    def test_auto_save_asset_default(self):
        """FALNode should have _auto_save_asset set to True by default."""
        assert FALNode._auto_save_asset is True
