import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nodetool.nodes.fal.fal_node import FALNode
from nodetool.nodes.fal.llm import OpenRouter
from nodetool.nodes.fal.text_to_image import (
    IdeogramV2,
    ImageSizePreset,
)


class TestNodeImports:
    """Test that all node modules can be imported correctly."""

    def test_import_llm_nodes(self):
        """Test that LLM nodes can be imported."""
        assert OpenRouter is not None
        assert issubclass(OpenRouter, FALNode)

    def test_import_text_to_image_nodes(self):
        """Test that text-to-image nodes can be imported."""
        assert IdeogramV2 is not None
        assert issubclass(IdeogramV2, FALNode)


class TestEnums:
    """Test enum definitions."""

    def test_image_size_preset_values(self):
        """Test ImageSizePreset contains expected values."""
        assert ImageSizePreset.SQUARE_HD.value == "square_hd"
        assert ImageSizePreset.PORTRAIT_16_9.value == "portrait_16_9"
        assert ImageSizePreset.LANDSCAPE_4_3.value == "landscape_4_3"

    def test_ideogram_aspect_ratio_values(self):
        """Test IdeogramV2 nested AspectRatio enum values."""
        assert IdeogramV2.AspectRatio.RATIO_1_1.value == "1:1"
        assert IdeogramV2.AspectRatio.RATIO_16_9.value == "16:9"
        assert IdeogramV2.AspectRatio.RATIO_9_16.value == "9:16"
        assert IdeogramV2.AspectRatio.RATIO_4_3.value == "4:3"

    def test_ideogram_style_values(self):
        """Test IdeogramV2 nested Style enum values."""
        assert IdeogramV2.Style.AUTO.value == "auto"
        assert IdeogramV2.Style.REALISTIC.value == "realistic"
        assert IdeogramV2.Style.ANIME.value == "anime"
        assert IdeogramV2.Style.RENDER_3D.value == "render_3D"


class TestNodeVisibility:
    """Test node visibility settings."""

    def test_openrouter_is_visible(self):
        """OpenRouter node should be visible."""
        assert OpenRouter.is_visible() is True

    def test_ideogram_v2_is_visible(self):
        """IdeogramV2 node should be visible."""
        assert IdeogramV2.is_visible() is True


class TestNodeBasicFields:
    """Test get_basic_fields method on nodes."""

    def test_openrouter_basic_fields(self):
        """Test OpenRouter basic fields."""
        fields = OpenRouter.get_basic_fields()
        assert "prompt" in fields
        assert "model" in fields

    def test_ideogram_v2_basic_fields(self):
        """Test IdeogramV2 basic fields."""
        if hasattr(IdeogramV2, "get_basic_fields"):
            fields = IdeogramV2.get_basic_fields()
            assert isinstance(fields, list)


class TestNewModuleImports:
    """Test that all new module categories can be imported."""

    def test_import_audio_to_text(self):
        from nodetool.nodes.fal import audio_to_text
        assert audio_to_text is not None

    def test_import_image_to_json(self):
        from nodetool.nodes.fal import image_to_json
        assert image_to_json is not None

    def test_import_json_processing(self):
        from nodetool.nodes.fal import json_processing
        assert json_processing is not None

    def test_import_speech_to_speech(self):
        from nodetool.nodes.fal import speech_to_speech
        assert speech_to_speech is not None

    def test_import_text_to_3d(self):
        from nodetool.nodes.fal import text_to_3d
        assert text_to_3d is not None

    def test_import_text_to_json(self):
        from nodetool.nodes.fal import text_to_json
        assert text_to_json is not None

    def test_import_text_to_text(self):
        from nodetool.nodes.fal import text_to_text
        assert text_to_text is not None

    def test_import_unknown(self):
        from nodetool.nodes.fal import unknown
        assert unknown is not None

    def test_import_video_to_audio(self):
        from nodetool.nodes.fal import video_to_audio
        assert video_to_audio is not None

    def test_import_video_to_text(self):
        from nodetool.nodes.fal import video_to_text
        assert video_to_text is not None


class TestKlingV3BaseTypes:
    """Test Kling V3 BaseType subclasses."""

    def test_multi_prompt_element_instantiation(self):
        from nodetool.nodes.fal.image_to_video import KlingV3MultiPromptElement
        mp = KlingV3MultiPromptElement(prompt="Test shot", duration="10")
        assert mp.prompt == "Test shot"
        assert mp.duration == "10"
        assert mp.type == "kling_v3_multi_prompt_element"

    def test_multi_prompt_element_defaults(self):
        from nodetool.nodes.fal.image_to_video import KlingV3MultiPromptElement
        mp = KlingV3MultiPromptElement()
        assert mp.prompt == ""
        assert mp.duration == "5"

    def test_multi_prompt_element_model_dump(self):
        from nodetool.nodes.fal.image_to_video import KlingV3MultiPromptElement
        mp = KlingV3MultiPromptElement(prompt="Hello", duration="7")
        d = mp.model_dump(exclude={"type"})
        assert d == {"prompt": "Hello", "duration": "7"}
        assert "type" not in d

    def test_combo_element_input_instantiation(self):
        from nodetool.nodes.fal.image_to_video import KlingV3ComboElementInput
        elem = KlingV3ComboElementInput(
            frontal_image_url="http://example.com/front.png",
            reference_image_urls=["http://example.com/ref1.png"],
            video_url=""
        )
        assert elem.frontal_image_url == "http://example.com/front.png"
        assert elem.reference_image_urls == ["http://example.com/ref1.png"]
        assert elem.type == "kling_v3_combo_element_input"

    def test_combo_element_input_defaults(self):
        from nodetool.nodes.fal.image_to_video import KlingV3ComboElementInput
        elem = KlingV3ComboElementInput()
        assert elem.frontal_image_url == ""
        assert elem.reference_image_urls == []
        assert elem.video_url == ""

    def test_combo_element_model_dump(self):
        from nodetool.nodes.fal.image_to_video import KlingV3ComboElementInput
        elem = KlingV3ComboElementInput(
            frontal_image_url="http://example.com/img.png",
            reference_image_urls=["http://example.com/ref.png"]
        )
        d = elem.model_dump(exclude={"type"})
        assert "frontal_image_url" in d
        assert "reference_image_urls" in d
        assert "type" not in d

    def test_base_type_from_dict(self):
        from nodetool.metadata.types import BaseType
        mp = BaseType.from_dict({
            "type": "kling_v3_multi_prompt_element",
            "prompt": "Shot 1",
            "duration": "3"
        })
        assert mp.prompt == "Shot 1"
        assert mp.duration == "3"


class TestKlingV3ImageToVideoNodes:
    """Test Kling V3 image-to-video node classes with advanced fields."""

    def test_standard_i2v_has_advanced_fields(self):
        from nodetool.nodes.fal.image_to_video import KlingVideoV3StandardImageToVideo
        node = KlingVideoV3StandardImageToVideo()
        assert hasattr(node, "voice_ids")
        assert hasattr(node, "multi_prompt")
        assert hasattr(node, "elements")
        assert hasattr(node, "shot_type")

    def test_standard_i2v_default_values(self):
        from nodetool.nodes.fal.image_to_video import KlingVideoV3StandardImageToVideo
        node = KlingVideoV3StandardImageToVideo()
        assert node.voice_ids == []
        assert node.multi_prompt == []
        assert node.elements == []
        assert node.shot_type == KlingVideoV3StandardImageToVideo.ShotType.CUSTOMIZE

    def test_standard_i2v_shot_type_enum(self):
        from nodetool.nodes.fal.image_to_video import KlingVideoV3StandardImageToVideo
        assert KlingVideoV3StandardImageToVideo.ShotType.CUSTOMIZE.value == "customize"

    def test_pro_i2v_has_advanced_fields(self):
        from nodetool.nodes.fal.image_to_video import KlingVideoV3ProImageToVideo
        node = KlingVideoV3ProImageToVideo()
        assert hasattr(node, "voice_ids")
        assert hasattr(node, "multi_prompt")
        assert hasattr(node, "elements")
        assert hasattr(node, "shot_type")

    def test_standard_i2v_with_multi_prompt(self):
        from nodetool.nodes.fal.image_to_video import (
            KlingVideoV3StandardImageToVideo,
            KlingV3MultiPromptElement
        )
        mp1 = KlingV3MultiPromptElement(prompt="Shot 1", duration="5")
        mp2 = KlingV3MultiPromptElement(prompt="Shot 2", duration="3")
        node = KlingVideoV3StandardImageToVideo(multi_prompt=[mp1, mp2])
        assert len(node.multi_prompt) == 2
        assert node.multi_prompt[0].prompt == "Shot 1"

    def test_standard_i2v_with_elements(self):
        from nodetool.nodes.fal.image_to_video import (
            KlingVideoV3StandardImageToVideo,
            KlingV3ComboElementInput
        )
        elem = KlingV3ComboElementInput(
            frontal_image_url="http://example.com/img.png",
            reference_image_urls=["http://example.com/ref.png"]
        )
        node = KlingVideoV3StandardImageToVideo(elements=[elem])
        assert len(node.elements) == 1

    def test_i2v_basic_fields(self):
        from nodetool.nodes.fal.image_to_video import KlingVideoV3StandardImageToVideo
        fields = KlingVideoV3StandardImageToVideo.get_basic_fields()
        assert "start_image_url" in fields
        assert "prompt" in fields
        assert "duration" in fields


class TestKlingV3TextToVideoNodes:
    """Test Kling V3 text-to-video node classes with advanced fields."""

    def test_standard_t2v_has_advanced_fields(self):
        from nodetool.nodes.fal.text_to_video import KlingVideoV3StandardTextToVideo
        node = KlingVideoV3StandardTextToVideo()
        assert hasattr(node, "voice_ids")
        assert hasattr(node, "multi_prompt")
        assert hasattr(node, "shot_type")

    def test_standard_t2v_no_elements(self):
        """Text-to-video should NOT have elements field (only i2v has it)."""
        from nodetool.nodes.fal.text_to_video import KlingVideoV3StandardTextToVideo
        node = KlingVideoV3StandardTextToVideo()
        assert not hasattr(node, "elements")

    def test_standard_t2v_shot_type_enum(self):
        from nodetool.nodes.fal.text_to_video import KlingVideoV3StandardTextToVideo
        assert KlingVideoV3StandardTextToVideo.ShotType.CUSTOMIZE.value == "customize"
        assert KlingVideoV3StandardTextToVideo.ShotType.INTELLIGENT.value == "intelligent"

    def test_pro_t2v_shot_type_enum(self):
        from nodetool.nodes.fal.text_to_video import KlingVideoV3ProTextToVideo
        assert KlingVideoV3ProTextToVideo.ShotType.CUSTOMIZE.value == "customize"
        assert KlingVideoV3ProTextToVideo.ShotType.INTELLIGENT.value == "intelligent"

    def test_t2v_with_multi_prompt(self):
        from nodetool.nodes.fal.text_to_video import KlingVideoV3StandardTextToVideo
        from nodetool.nodes.fal.image_to_video import KlingV3MultiPromptElement
        mp = KlingV3MultiPromptElement(prompt="Scene 1", duration="5")
        node = KlingVideoV3StandardTextToVideo(
            multi_prompt=[mp],
            shot_type=KlingVideoV3StandardTextToVideo.ShotType.INTELLIGENT
        )
        assert len(node.multi_prompt) == 1
        assert node.shot_type.value == "intelligent"

    def test_t2v_basic_fields(self):
        from nodetool.nodes.fal.text_to_video import KlingVideoV3StandardTextToVideo
        fields = KlingVideoV3StandardTextToVideo.get_basic_fields()
        assert "prompt" in fields
        assert "duration" in fields
        assert "aspect_ratio" in fields


class TestSchemaParserRefResolution:
    """Test that the schema parser correctly resolves $ref types in arrays."""

    def test_resolve_ref_in_array_items(self):
        import sys
        sys.path.insert(0, ".")
        from codegen.schema_parser import SchemaParser

        schema = {
            "info": {"x-fal-metadata": {"endpointId": "test/endpoint"}},
            "components": {
                "schemas": {
                    "TestInput": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {"$ref": "#/components/schemas/MyElement"}
                            }
                        }
                    },
                    "MyElement": {
                        "title": "MyElement",
                        "type": "object",
                        "properties": {"name": {"type": "string"}}
                    },
                    "TestOutput": {
                        "type": "object",
                        "properties": {
                            "video": {
                                "allOf": [{"$ref": "#/components/schemas/File"}]
                            }
                        },
                        "required": ["video"]
                    },
                    "File": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"]
                    }
                }
            },
            "paths": {
                "/test/endpoint": {
                    "post": {
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/TestInput"}
                                }
                            }
                        }
                    }
                },
                "/test/endpoint/requests/{request_id}": {
                    "get": {
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/TestOutput"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        parser = SchemaParser()
        spec = parser.parse(schema)
        items_field = next(f for f in spec.input_fields if f.name == "items")
        assert items_field.python_type == "list[MyElement]"

    def test_plain_array_still_works(self):
        import sys
        sys.path.insert(0, ".")
        from codegen.schema_parser import SchemaParser

        schema = {
            "info": {"x-fal-metadata": {"endpointId": "test/endpoint2"}},
            "components": {
                "schemas": {
                    "TestInput": {
                        "type": "object",
                        "properties": {
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        }
                    },
                    "TestOutput": {
                        "type": "object",
                        "properties": {
                            "video": {
                                "allOf": [{"$ref": "#/components/schemas/File"}]
                            }
                        },
                        "required": ["video"]
                    },
                    "File": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"]
                    }
                }
            },
            "paths": {
                "/test/endpoint2": {
                    "post": {
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/TestInput"}
                                }
                            }
                        }
                    }
                },
                "/test/endpoint2/requests/{request_id}": {
                    "get": {
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/TestOutput"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        parser = SchemaParser()
        spec = parser.parse(schema)
        tags_field = next(f for f in spec.input_fields if f.name == "tags")
        assert tags_field.python_type == "list[str]"
