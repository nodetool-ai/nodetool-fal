import os
import sys
from unittest.mock import Mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nodetool.nodes.fal.fal_node import FALNode
from nodetool.providers.cost_calculator import UsageInfo


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

    def test_track_cost_with_images(self):
        """Test cost tracking with image outputs."""

        class TestNode(FALNode):
            pass

        node = TestNode()
        # Use object.__setattr__ to bypass the property setter restriction
        object.__setattr__(node, "_id", "test-node-123")

        context = Mock()
        context.track_operation_cost = Mock()

        # Test with multiple images
        result = {"images": [{"url": "http://example.com/1.png"}] * 3}
        inference_time = 2.5

        node._track_cost(context, "fal-ai/flux/dev", result, inference_time)

        # Verify track_operation_cost was called correctly
        context.track_operation_cost.assert_called_once()
        call_args = context.track_operation_cost.call_args
        assert call_args.kwargs["model"] == "fal-ai/flux/dev"
        assert call_args.kwargs["provider"] == "fal"
        assert call_args.kwargs["node_id"] == "test-node-123"
        assert call_args.kwargs["operation_type"] == "prediction"

        # Check usage_info
        usage_info = call_args.kwargs["usage_info"]
        assert isinstance(usage_info, UsageInfo)
        assert usage_info.duration_seconds == 2.5
        assert usage_info.image_count == 3
        assert usage_info.video_seconds == 0.0

    def test_track_cost_with_single_image(self):
        """Test cost tracking with single image output."""

        class TestNode(FALNode):
            pass

        node = TestNode()
        object.__setattr__(node, "_id", "test-node-456")

        context = Mock()
        context.track_operation_cost = Mock()

        # Test with single image
        result = {"image": {"url": "http://example.com/image.png"}}
        inference_time = 1.0

        node._track_cost(context, "fal-ai/sdxl", result, inference_time)

        # Verify the call
        context.track_operation_cost.assert_called_once()
        usage_info = context.track_operation_cost.call_args.kwargs["usage_info"]
        assert usage_info.image_count == 1

    def test_track_cost_with_video(self):
        """Test cost tracking with video output."""

        class TestNode(FALNode):
            pass

        node = TestNode()
        object.__setattr__(node, "_id", "test-node-789")

        context = Mock()
        context.track_operation_cost = Mock()

        # Test with video
        result = {"video": {"url": "http://example.com/video.mp4"}}
        inference_time = 5.0

        node._track_cost(context, "fal-ai/video-gen", result, inference_time)

        # Verify the call
        context.track_operation_cost.assert_called_once()
        usage_info = context.track_operation_cost.call_args.kwargs["usage_info"]
        assert usage_info.duration_seconds == 5.0
        assert usage_info.video_seconds == 0.0  # Video duration not in response
        assert usage_info.image_count == 0

    def test_track_cost_with_no_outputs(self):
        """Test cost tracking with no image or video outputs."""

        class TestNode(FALNode):
            pass

        node = TestNode()
        object.__setattr__(node, "_id", "test-node-abc")

        context = Mock()
        context.track_operation_cost = Mock()

        # Test with no recognized outputs
        result = {"text": "some result"}
        inference_time = 0.5

        node._track_cost(context, "fal-ai/text-model", result, inference_time)

        # Verify the call
        context.track_operation_cost.assert_called_once()
        usage_info = context.track_operation_cost.call_args.kwargs["usage_info"]
        assert usage_info.duration_seconds == 0.5
        assert usage_info.image_count == 0
        assert usage_info.video_seconds == 0.0
