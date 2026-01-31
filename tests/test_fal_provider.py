import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nodetool.fal.fal_provider import FalProvider


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
