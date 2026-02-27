import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nodetool.metadata.types import VideoRef
from nodetool.nodes.fal.dynamic_schema import (
    FalAI,
    _build_output_types,
    _default_value_for_input_property,
    _map_output_values,
    _normalize_model_info,
    _parse_openapi_schema,
    _parse_model_info_text,
    _schema_bundle_to_resolve_result,
)

SCHEMA_EXAMPLE = {
    "openapi": "3.0.4",
    "info": {
        "title": "Queue OpenAPI for fal-ai/kling-video/o3/standard/image-to-video",
        "version": "1.0.0",
        "x-fal-metadata": {
            "endpointId": "fal-ai/kling-video/o3/standard/image-to-video",
        },
    },
    "components": {
        "schemas": {
            "QueueStatus": {
                "type": "object",
                "properties": {"status": {"type": "string"}},
                "required": ["status"],
            },
            "KlingVideoO3StandardImageToVideoInput": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "image_url": {"type": "string"},
                },
                "required": ["image_url"],
            },
            "KlingVideoO3StandardImageToVideoOutput": {
                "type": "object",
                "properties": {
                    "video": {
                        "allOf": [
                            {"$ref": "#/components/schemas/File"},
                        ]
                    }
                },
                "required": ["video"],
            },
            "File": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        }
    },
    "paths": {
        "/fal-ai/kling-video/o3/standard/image-to-video": {
            "post": {
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/KlingVideoO3StandardImageToVideoInput"
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/QueueStatus"}
                            }
                        }
                    }
                },
            }
        },
        "/fal-ai/kling-video/o3/standard/image-to-video/requests/{request_id}": {
            "get": {
                "responses": {
                    "200": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/KlingVideoO3StandardImageToVideoOutput"
                                }
                            }
                        }
                    }
                }
            }
        },
    },
}


SCHEMA_WITH_ALLOF_INPUTS = {
    "openapi": "3.0.4",
    "info": {
        "title": "Queue OpenAPI for fal-ai/nano-banana-2",
        "version": "1.0.0",
        "x-fal-metadata": {"endpointId": "fal-ai/nano-banana-2"},
    },
    "components": {
        "schemas": {
            "NanoBanana2Input": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "allOf": [{"$ref": "#/components/schemas/PromptField"}]
                    },
                    "seed": {"allOf": [{"$ref": "#/components/schemas/SeedField"}]},
                    "aspect_ratio": {
                        "allOf": [
                            {"$ref": "#/components/schemas/AspectRatioField"}
                        ]
                    },
                },
                "required": ["prompt"],
            },
            "PromptField": {"type": "string"},
            "SeedField": {"type": "integer"},
            "AspectRatioField": {
                "type": "string",
                "default": "auto",
                "enum": ["auto", "16:9", "1:1"],
            },
            "NanoBanana2Output": {
                "type": "object",
                "properties": {"description": {"type": "string"}},
                "required": ["description"],
            },
        }
    },
    "paths": {
        "/fal-ai/nano-banana-2": {
            "post": {
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/NanoBanana2Input"}
                        }
                    },
                },
                "responses": {"200": {"description": "ok"}},
            }
        },
        "/fal-ai/nano-banana-2/requests/{request_id}": {
            "get": {
                "responses": {
                    "200": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/NanoBanana2Output"}
                            }
                        }
                    }
                }
            }
        },
    },
}


SCHEMA_WITH_WRAPPER_DEFAULTS = {
    "openapi": "3.0.4",
    "info": {
        "title": "Queue OpenAPI for fal-ai/nano-banana-2",
        "version": "1.0.0",
        "x-fal-metadata": {"endpointId": "fal-ai/nano-banana-2"},
    },
    "components": {
        "schemas": {
            "Input": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "allOf": [{"$ref": "#/components/schemas/PromptType"}],
                    },
                    "resolution": {
                        "allOf": [{"$ref": "#/components/schemas/ResolutionType"}],
                        "default": "1K",
                    },
                },
                "required": ["prompt"],
            },
            "PromptType": {"type": "string"},
            "ResolutionType": {
                "type": "string",
                "enum": ["0.5K", "1K", "2K", "4K"],
            },
            "Output": {
                "type": "object",
                "properties": {"description": {"type": "string"}},
                "required": ["description"],
            },
        }
    },
    "paths": {
        "/fal-ai/nano-banana-2": {
            "post": {
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Input"}
                        }
                    },
                },
                "responses": {"200": {"description": "ok"}},
            }
        },
        "/fal-ai/nano-banana-2/requests/{request_id}": {
            "get": {
                "responses": {
                    "200": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Output"}
                            }
                        }
                    }
                }
            }
        },
    },
}


SCHEMA_OUTPUT_WITH_DESCRIPTION = {
    "openapi": "3.0.4",
    "info": {
        "title": "Queue OpenAPI for fal-ai/test-output",
        "version": "1.0.0",
        "x-fal-metadata": {"endpointId": "fal-ai/test-output"},
    },
    "components": {
        "schemas": {
            "Input": {
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
                "required": ["prompt"],
            },
            "File": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            "Output": {
                "type": "object",
                "properties": {
                    "images": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/File"},
                    },
                    "description": {"type": "string"},
                },
                "required": ["images", "description"],
            },
        }
    },
    "paths": {
        "/fal-ai/test-output": {
            "post": {
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Input"}
                        }
                    },
                },
                "responses": {"200": {"description": "ok"}},
            }
        },
        "/fal-ai/test-output/requests/{request_id}": {
            "get": {
                "responses": {
                    "200": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Output"}
                            }
                        }
                    }
                }
            }
        },
    },
}


def test_model_info_normalization():
    model_url = "https://fal.ai/models/fal-ai/kling-video/o3/standard/image-to-video"
    text, url, endpoint = _normalize_model_info(model_url)
    assert text is None
    assert url.endswith("/llms.txt")
    assert endpoint == "fal-ai/kling-video/o3/standard/image-to-video"

    text, url, endpoint = _normalize_model_info(
        "fal-ai/kling-video/o3/standard/image-to-video"
    )
    assert text is None
    assert url.endswith("/llms.txt")
    assert endpoint == "fal-ai/kling-video/o3/standard/image-to-video"


def test_parse_model_info_text():
    model_info = (
        "- **Endpoint**: `https://fal.run/fal-ai/kling-video/o3/standard/image-to-video`\n"
        "- **Model ID**: `fal-ai/kling-video/o3/standard/image-to-video`\n"
        "- **OpenAPI Schema**: "
        "`https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=fal-ai/kling-video/o3/standard/image-to-video`\n"
    )
    endpoint_id, openapi_url = _parse_model_info_text(model_info, None)
    assert endpoint_id == "fal-ai/kling-video/o3/standard/image-to-video"
    assert openapi_url.endswith(
        "endpoint_id=fal-ai/kling-video/o3/standard/image-to-video"
    )


def test_parse_openapi_schema():
    bundle = _parse_openapi_schema(SCHEMA_EXAMPLE, llm_info="metadata")
    assert bundle.endpoint_id == "fal-ai/kling-video/o3/standard/image-to-video"
    assert "image_url" in bundle.input_schema["properties"]
    assert "video" in bundle.output_schema["properties"]


def test_output_types_and_mapping():
    bundle = _parse_openapi_schema(SCHEMA_EXAMPLE, llm_info=None)
    output_types = _build_output_types(bundle.openapi, bundle.output_schema)
    assert output_types["video"].type == "video"

    response = {
        "video": {
            "url": "https://example.com/video.mp4",
            "file_name": "output.mp4",
        }
    }
    outputs = _map_output_values(bundle.openapi, bundle.output_schema, response)
    assert isinstance(outputs["video"], VideoRef)
    assert outputs["video"].uri == "https://example.com/video.mp4"


def test_description_output_is_omitted_for_dynamic_fal():
    bundle = _parse_openapi_schema(SCHEMA_OUTPUT_WITH_DESCRIPTION, llm_info=None)

    output_types = _build_output_types(bundle.openapi, bundle.output_schema)
    assert "description" not in output_types
    assert "images" in output_types

    response = {
        "images": [{"url": "https://example.com/image.png"}],
        "description": "Generated image",
    }
    outputs = _map_output_values(bundle.openapi, bundle.output_schema, response)
    assert "description" not in outputs
    assert "images" in outputs


@pytest.mark.asyncio
async def test_build_arguments_skips_none():
    bundle = _parse_openapi_schema(SCHEMA_EXAMPLE, llm_info=None)
    node = FalAI()
    arguments = await node._build_arguments(
        bundle.openapi,
        bundle.input_schema,
        {"image_url": "https://example.com/start.png", "prompt": None, "extra": "skip"},
        context=None,
    )
    assert arguments == {"image_url": "https://example.com/start.png"}


@pytest.mark.asyncio
async def test_build_arguments_omits_optional_seed_zero():
    schema = {
        "openapi": "3.0.4",
        "info": {"x-fal-metadata": {"endpointId": "fal-ai/test"}},
        "components": {
            "schemas": {
                "Input": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "seed": {"type": "integer"},
                    },
                    "required": ["prompt"],
                },
                "Output": {
                    "type": "object",
                    "properties": {"description": {"type": "string"}},
                    "required": ["description"],
                },
            }
        },
        "paths": {
            "/fal-ai/test": {
                "post": {
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Input"}
                            }
                        },
                    },
                    "responses": {"200": {"description": "ok"}},
                }
            },
            "/fal-ai/test/requests/{request_id}": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Output"}
                                }
                            }
                        }
                    }
                }
            },
        },
    }
    bundle = _parse_openapi_schema(schema, llm_info=None)
    node = FalAI()
    arguments = await node._build_arguments(
        bundle.openapi,
        bundle.input_schema,
        {"prompt": "hello", "seed": 0},
        context=None,
    )
    assert arguments == {"prompt": "hello"}


@pytest.mark.asyncio
async def test_build_arguments_omits_optional_seed_minus_one():
    schema = {
        "openapi": "3.0.4",
        "info": {"x-fal-metadata": {"endpointId": "fal-ai/test"}},
        "components": {
            "schemas": {
                "Input": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "seed": {"type": "integer"},
                    },
                    "required": ["prompt"],
                },
                "Output": {
                    "type": "object",
                    "properties": {"description": {"type": "string"}},
                    "required": ["description"],
                },
            }
        },
        "paths": {
            "/fal-ai/test": {
                "post": {
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Input"}
                            }
                        },
                    },
                    "responses": {"200": {"description": "ok"}},
                }
            },
            "/fal-ai/test/requests/{request_id}": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Output"}
                                }
                            }
                        }
                    }
                }
            },
        },
    }
    bundle = _parse_openapi_schema(schema, llm_info=None)
    node = FalAI()
    arguments = await node._build_arguments(
        bundle.openapi,
        bundle.input_schema,
        {"prompt": "hello", "seed": -1},
        context=None,
    )
    assert arguments == {"prompt": "hello"}


def test_allof_scalar_inputs_preserve_type_and_defaults():
    bundle = _parse_openapi_schema(SCHEMA_WITH_ALLOF_INPUTS, llm_info=None)
    props = bundle.input_schema["properties"]
    required = set(bundle.input_schema.get("required", []))

    prompt_default = _default_value_for_input_property(
        bundle.openapi, props["prompt"], required=("prompt" in required), prop_name="prompt"
    )
    aspect_default = _default_value_for_input_property(
        bundle.openapi,
        props["aspect_ratio"],
        required=("aspect_ratio" in required),
        prop_name="aspect_ratio",
    )
    seed_default = _default_value_for_input_property(
        bundle.openapi, props["seed"], required=("seed" in required), prop_name="seed"
    )

    assert prompt_default == ""
    assert aspect_default == "auto"
    assert seed_default == -1


def test_wrapper_level_defaults_are_preserved_for_allof_fields():
    bundle = _parse_openapi_schema(SCHEMA_WITH_WRAPPER_DEFAULTS, llm_info=None)
    props = bundle.input_schema["properties"]
    required = set(bundle.input_schema.get("required", []))

    prompt_default = _default_value_for_input_property(
        bundle.openapi, props["prompt"], required=("prompt" in required), prop_name="prompt"
    )
    resolution_default = _default_value_for_input_property(
        bundle.openapi,
        props["resolution"],
        required=("resolution" in required),
        prop_name="resolution",
    )

    assert prompt_default == ""
    assert resolution_default == "1K"


def test_resolve_result_includes_effective_defaults_for_dynamic_inputs():
    bundle = _parse_openapi_schema(SCHEMA_WITH_ALLOF_INPUTS, llm_info=None)
    resolved = _schema_bundle_to_resolve_result(bundle)
    dynamic_inputs = resolved["dynamic_inputs"]

    assert dynamic_inputs["prompt"]["default"] == ""
    assert dynamic_inputs["seed"]["default"] == -1
