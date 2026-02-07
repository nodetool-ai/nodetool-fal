import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nodetool.nodes.fal.speech_to_text import (
    ElevenLabsScribeV2,
    Whisper,
)
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.metadata.types import AudioRef


class TestSpeechToTextNodeImports:
    """Test that speech-to-text nodes can be imported correctly."""

    def test_import_elevenlabs_scribe_v2(self):
        """Test that ElevenLabsScribeV2 node can be imported."""
        assert ElevenLabsScribeV2 is not None
        assert issubclass(ElevenLabsScribeV2, FALNode)

    def test_import_whisper(self):
        """Test that Whisper node can be imported."""
        assert Whisper is not None
        assert issubclass(Whisper, FALNode)


class TestSpeechToTextNodeVisibility:
    """Test node visibility settings for speech-to-text."""

    def test_elevenlabs_scribe_v2_is_visible(self):
        """ElevenLabsScribeV2 node should be visible."""
        assert ElevenLabsScribeV2.is_visible() is True

    def test_whisper_is_visible(self):
        """Whisper node should be visible."""
        assert Whisper.is_visible() is True


class TestSpeechToTextNodeInstantiation:
    """Test that speech-to-text nodes can be instantiated with default values."""

    def test_elevenlabs_scribe_v2_instantiation(self):
        """Test ElevenLabsScribeV2 node instantiation."""
        node = ElevenLabsScribeV2()
        assert isinstance(node.audio, AudioRef)

    def test_whisper_instantiation(self):
        """Test Whisper node instantiation."""
        node = Whisper()
        assert isinstance(node.audio, AudioRef)
        assert node.task == "transcribe"


class TestSpeechToTextBasicFields:
    """Test get_basic_fields method on speech-to-text nodes."""

    def test_elevenlabs_scribe_v2_basic_fields(self):
        """Test ElevenLabsScribeV2 basic fields."""
        if hasattr(ElevenLabsScribeV2, "get_basic_fields"):
            fields = ElevenLabsScribeV2.get_basic_fields()
            assert isinstance(fields, list)
            assert len(fields) > 0

    def test_whisper_basic_fields(self):
        """Test Whisper basic fields."""
        if hasattr(Whisper, "get_basic_fields"):
            fields = Whisper.get_basic_fields()
            assert isinstance(fields, list)
            assert len(fields) > 0


class TestSpeechToTextReturnTypes:
    """Test return types for speech-to-text nodes."""

    def test_whisper_return_type(self):
        """Test Whisper return type."""
        if hasattr(Whisper, "return_type"):
            return_type = Whisper.return_type()
            assert return_type is not None
