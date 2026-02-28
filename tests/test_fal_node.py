import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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

    @pytest.mark.asyncio
    async def test_submit_request_filters_empty_strings(self):
        """Empty string values should be filtered from arguments to avoid API parsing errors."""
        node = FALNode()
        context = MagicMock()

        async def fake_aiter(*args, **kwargs):
            if False:
                yield  # make it an async generator

        mock_handler = MagicMock()
        mock_handler.iter_events = MagicMock(return_value=fake_aiter())
        mock_handler.get = AsyncMock(
            return_value={"images": [{"url": "http://example.com"}]}
        )

        mock_client = AsyncMock()
        mock_client.submit = AsyncMock(return_value=mock_handler)

        with patch.object(
            FALNode, "get_client", new_callable=AsyncMock, return_value=mock_client
        ):
            await node.submit_request(
                context=context,
                application="fal-ai/test",
                arguments={
                    "prompt": "test prompt",
                    "seed": "",
                    "negative_prompt": "",
                    "guidance_scale": 3.5,
                },
            )

        # Verify that empty string values were filtered out
        call_args = mock_client.submit.call_args
        submitted_args = call_args.kwargs.get(
            "arguments", call_args[1].get("arguments")
        )
        assert "seed" not in submitted_args
        assert "negative_prompt" not in submitted_args
        assert submitted_args["prompt"] == "test prompt"
        assert submitted_args["guidance_scale"] == 3.5
