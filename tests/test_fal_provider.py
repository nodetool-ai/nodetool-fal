import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nodetool.fal.fal_provider import (
    FalProvider,
    _get_discovered_nodes,
    _find_node_class,
    _get_endpoint_id,
    _get_node_name,
    _node_class_path,
)


class TestFalProviderValidationErrors:
    """Tests for FalProvider._format_validation_error method."""

    def test_format_less_than_equal_error(self):
        """Test formatting of less_than_equal validation errors."""
        error_str = "[{'type': 'less_than_equal', 'loc': ['body', 'num_inference_steps'], 'msg': 'Input should be less than or equal to 12', 'input': 30, 'ctx': {'le': 12}}]"
        result = FalProvider._format_validation_error(error_str)
        assert "num_inference_steps" in result
        assert "12" in result
        assert "30" in result

    def test_format_greater_than_equal_error(self):
        """Test formatting of greater_than_equal validation errors."""
        error_str = "[{'type': 'greater_than_equal', 'loc': ['body', 'width'], 'msg': 'Input should be greater than or equal to 64', 'input': 10, 'ctx': {'ge': 64}}]"
        result = FalProvider._format_validation_error(error_str)
        assert "width" in result
        assert "64" in result
        assert "10" in result

    def test_format_missing_field_error(self):
        """Test formatting of missing field errors."""
        error_str = "[{'type': 'missing', 'loc': ['body', 'prompt'], 'msg': 'Field required', 'input': None}]"
        result = FalProvider._format_validation_error(error_str)
        assert "prompt" in result
        assert "missing" in result

    def test_format_value_error(self):
        """Test formatting of value errors."""
        error_str = "[{'type': 'value_error', 'loc': ['body', 'seed'], 'msg': 'Value must be positive', 'input': -5}]"
        result = FalProvider._format_validation_error(error_str)
        assert "seed" in result
        assert "Value must be positive" in result

    def test_format_generic_error_with_input(self):
        """Test formatting of generic errors with input value."""
        error_str = "[{'type': 'string_too_long', 'loc': ['body', 'prompt'], 'msg': 'String too long', 'input': 'very long text'}]"
        result = FalProvider._format_validation_error(error_str)
        assert "prompt" in result
        assert "String too long" in result

    def test_format_unparseable_error(self):
        """Test that unparseable errors are returned as-is."""
        error_str = "Some random error message"
        result = FalProvider._format_validation_error(error_str)
        assert result == error_str

    def test_format_empty_error_list(self):
        """Test handling of empty error list."""
        error_str = "[]"
        result = FalProvider._format_validation_error(error_str)
        # Should return original since there's nothing to format
        assert result == error_str

    def test_format_json_formatted_error(self):
        """Test formatting of JSON-formatted errors (double quotes)."""
        import json

        error_list = [
            {
                "type": "less_than_equal",
                "loc": ["body", "steps"],
                "msg": "Too high",
                "input": 100,
                "ctx": {"le": 50},
            }
        ]
        error_str = f"Validation error: {json.dumps(error_list)}"
        result = FalProvider._format_validation_error(error_str)
        assert "steps" in result
        assert "50" in result
        assert "100" in result

    def test_format_multiple_errors(self):
        """Test formatting of multiple validation errors."""
        error_str = "[{'type': 'missing', 'loc': ['body', 'prompt'], 'msg': 'Field required', 'input': None}, {'type': 'less_than_equal', 'loc': ['body', 'steps'], 'msg': 'Too high', 'input': 100, 'ctx': {'le': 50}}]"
        result = FalProvider._format_validation_error(error_str)
        assert "prompt" in result
        assert "steps" in result


class TestFalProviderRequiredSecrets:
    """Tests for FalProvider.required_secrets method."""

    def test_required_secrets(self):
        """Test that required secrets include FAL_API_KEY."""
        secrets = FalProvider.required_secrets()
        assert "FAL_API_KEY" in secrets
        assert len(secrets) == 1


class TestNodeDiscovery:
    """Tests for dynamic FAL node discovery."""

    def test_discover_returns_all_categories(self):
        """Discovered nodes should include all expected categories."""
        nodes = _get_discovered_nodes()
        assert "image" in nodes
        assert "video" in nodes
        assert "tts" in nodes
        assert "asr" in nodes
        assert "3d" in nodes

    def test_discover_image_nodes_not_empty(self):
        """Image category should contain many discovered nodes."""
        nodes = _get_discovered_nodes()
        assert len(nodes["image"]) > 50

    def test_discover_video_nodes_not_empty(self):
        """Video category should contain discovered nodes."""
        nodes = _get_discovered_nodes()
        assert len(nodes["video"]) > 10

    def test_discover_tts_nodes_not_empty(self):
        """TTS category should contain discovered nodes."""
        nodes = _get_discovered_nodes()
        assert len(nodes["tts"]) > 5

    def test_discover_asr_nodes_not_empty(self):
        """ASR category should contain discovered nodes."""
        nodes = _get_discovered_nodes()
        assert len(nodes["asr"]) > 2

    def test_discover_3d_nodes_not_empty(self):
        """3D category should contain discovered nodes."""
        nodes = _get_discovered_nodes()
        assert len(nodes["3d"]) > 5

    def test_discovered_node_tuples_format(self):
        """Each discovered node should be a (endpoint, name, class_path) tuple."""
        nodes = _get_discovered_nodes()
        for cat, items in nodes.items():
            for item in items:
                assert len(item) == 3
                endpoint, name, class_path = item
                assert isinstance(endpoint, str)
                assert isinstance(name, str)
                assert isinstance(class_path, str)
                assert "." in class_path  # fully qualified

    def test_discover_known_flux_dev(self):
        """Should discover the well-known fal-ai/flux/dev endpoint."""
        nodes = _get_discovered_nodes()
        endpoints = [ep for ep, _, _ in nodes["image"]]
        assert "fal-ai/flux/dev" in endpoints

    def test_discover_known_hunyuan_video(self):
        """Should discover the well-known fal-ai/hunyuan-video endpoint."""
        nodes = _get_discovered_nodes()
        endpoints = [ep for ep, _, _ in nodes["video"]]
        assert "fal-ai/hunyuan-video" in endpoints


class TestFindNodeClass:
    """Tests for _find_node_class lookup."""

    def test_find_flux_dev(self):
        """Should find the FluxDev node class."""
        cls = _find_node_class("fal-ai/flux/dev", category="image")
        assert cls is not None
        assert cls.__name__ == "FluxDev"

    def test_find_hunyuan_video(self):
        """Should find the HunyuanVideo node class."""
        cls = _find_node_class("fal-ai/hunyuan-video", category="video")
        assert cls is not None
        assert cls.__name__ == "HunyuanVideo"

    def test_find_trellis(self):
        """Should find the Trellis node class."""
        cls = _find_node_class("fal-ai/trellis", category="3d")
        assert cls is not None
        assert cls.__name__ == "Trellis"

    def test_find_without_category(self):
        """Should find a node class even without specifying category."""
        cls = _find_node_class("fal-ai/flux/dev")
        assert cls is not None
        assert cls.__name__ == "FluxDev"

    def test_find_nonexistent_returns_none(self):
        """Should return None for unknown endpoint IDs."""
        cls = _find_node_class("fal-ai/nonexistent-model-xyz")
        assert cls is None


class TestNodeIntrospectionHelpers:
    """Tests for endpoint extraction and naming helpers."""

    def test_get_endpoint_id_from_class(self):
        """Should extract endpoint ID from a FAL node class."""
        from nodetool.nodes.fal.text_to_image import FluxDev

        endpoint = _get_endpoint_id(FluxDev)
        assert endpoint == "fal-ai/flux/dev"

    def test_get_node_name_from_docstring(self):
        """Should derive model name from docstring."""
        from nodetool.nodes.fal.text_to_image import FluxDev

        name = _get_node_name(FluxDev)
        assert "FLUX" in name

    def test_node_class_path(self):
        """Should return fully qualified class path."""
        from nodetool.nodes.fal.text_to_image import FluxDev

        path = _node_class_path(FluxDev)
        assert path == "nodetool.nodes.fal.text_to_image.FluxDev"


class TestProviderCapabilities:
    """Tests for FalProvider capability detection."""

    def test_capabilities_include_all_types(self):
        """Provider should report all implemented capabilities."""
        os.environ["FAL_API_KEY"] = "test-key"
        provider = FalProvider(secrets={"FAL_API_KEY": "test-key"})
        caps = provider.get_capabilities()
        cap_names = {str(c) for c in caps}

        assert "ProviderCapability.TEXT_TO_IMAGE" in cap_names
        assert "ProviderCapability.IMAGE_TO_IMAGE" in cap_names
        assert "ProviderCapability.TEXT_TO_SPEECH" in cap_names
        assert "ProviderCapability.TEXT_TO_VIDEO" in cap_names
        assert "ProviderCapability.IMAGE_TO_VIDEO" in cap_names
        assert "ProviderCapability.TEXT_TO_3D" in cap_names
        assert "ProviderCapability.IMAGE_TO_3D" in cap_names
        assert "ProviderCapability.AUTOMATIC_SPEECH_RECOGNITION" in cap_names


class TestProviderModelListing:
    """Tests for dynamic model listing via get_available_*_models."""

    @pytest.fixture
    def provider(self):
        os.environ["FAL_API_KEY"] = "test-key"
        return FalProvider(secrets={"FAL_API_KEY": "test-key"})

    @pytest.mark.asyncio
    async def test_get_available_image_models(self, provider):
        """Should return dynamically discovered image models."""
        models = await provider.get_available_image_models()
        assert len(models) > 50
        ids = [m.id for m in models]
        assert "fal-ai/flux/dev" in ids
        # Each model should have a path for node lookup
        for m in models:
            assert m.path is not None

    @pytest.mark.asyncio
    async def test_get_available_video_models(self, provider):
        """Should return dynamically discovered video models."""
        models = await provider.get_available_video_models()
        assert len(models) > 10
        ids = [m.id for m in models]
        assert "fal-ai/hunyuan-video" in ids

    @pytest.mark.asyncio
    async def test_get_available_tts_models(self, provider):
        """Should return dynamically discovered TTS models."""
        models = await provider.get_available_tts_models()
        assert len(models) > 5

    @pytest.mark.asyncio
    async def test_get_available_asr_models(self, provider):
        """Should return dynamically discovered ASR models."""
        models = await provider.get_available_asr_models()
        assert len(models) > 2

    @pytest.mark.asyncio
    async def test_get_available_3d_models(self, provider):
        """Should return dynamically discovered 3D models."""
        models = await provider.get_available_3d_models()
        assert len(models) > 5
