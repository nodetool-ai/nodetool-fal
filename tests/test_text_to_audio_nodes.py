import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nodetool.nodes.fal.text_to_audio import (
    StableAudio,
    ElevenLabsTTSV3,
    ElevenLabsSoundEffects,
)
from nodetool.nodes.fal.fal_node import FALNode


class TestTextToAudioNodeImports:
    """Test that text-to-audio nodes can be imported correctly."""

    def test_import_stable_audio(self):
        """Test that StableAudio node can be imported."""
        assert StableAudio is not None
        assert issubclass(StableAudio, FALNode)

    def test_import_elevenlabs_tts(self):
        """Test that ElevenLabsTTSV3 node can be imported."""
        assert ElevenLabsTTSV3 is not None
        assert issubclass(ElevenLabsTTSV3, FALNode)

    def test_import_elevenlabs_sound_effects(self):
        """Test that ElevenLabsSoundEffects node can be imported."""
        assert ElevenLabsSoundEffects is not None
        assert issubclass(ElevenLabsSoundEffects, FALNode)


class TestTextToAudioNodeVisibility:
    """Test node visibility settings for text-to-audio."""

    def test_stable_audio_is_visible(self):
        """StableAudio node should be visible."""
        assert StableAudio.is_visible() is True

    def test_elevenlabs_tts_is_visible(self):
        """ElevenLabsTTSV3 node should be visible."""
        assert ElevenLabsTTSV3.is_visible() is True

    def test_elevenlabs_sound_effects_is_visible(self):
        """ElevenLabsSoundEffects node should be visible."""
        assert ElevenLabsSoundEffects.is_visible() is True


class TestTextToAudioNodeInstantiation:
    """Test that text-to-audio nodes can be instantiated with default values."""

    def test_stable_audio_instantiation(self):
        """Test StableAudio node instantiation."""
        node = StableAudio()
        assert node.prompt == ""

    def test_elevenlabs_tts_instantiation(self):
        """Test ElevenLabsTTSV3 node instantiation."""
        node = ElevenLabsTTSV3()
        assert node.text == ""

    def test_elevenlabs_sound_effects_instantiation(self):
        """Test ElevenLabsSoundEffects node instantiation."""
        node = ElevenLabsSoundEffects()
        assert node.prompt == ""
        assert node.duration == 5.0


class TestTextToAudioBasicFields:
    """Test get_basic_fields method on text-to-audio nodes."""

    def test_stable_audio_basic_fields(self):
        """Test StableAudio basic fields."""
        if hasattr(StableAudio, "get_basic_fields"):
            fields = StableAudio.get_basic_fields()
            assert isinstance(fields, list)
            assert len(fields) > 0

    def test_elevenlabs_tts_basic_fields(self):
        """Test ElevenLabsTTSV3 basic fields."""
        if hasattr(ElevenLabsTTSV3, "get_basic_fields"):
            fields = ElevenLabsTTSV3.get_basic_fields()
            assert isinstance(fields, list)
            assert len(fields) > 0

    def test_elevenlabs_sound_effects_basic_fields(self):
        """Test ElevenLabsSoundEffects basic fields."""
        if hasattr(ElevenLabsSoundEffects, "get_basic_fields"):
            fields = ElevenLabsSoundEffects.get_basic_fields()
            assert isinstance(fields, list)
            assert len(fields) > 0
