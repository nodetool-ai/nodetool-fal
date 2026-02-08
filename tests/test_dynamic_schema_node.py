import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nodetool.metadata.types import VideoRef
from nodetool.nodes.fal.dynamic_schema import (
    FalAI,
    _build_output_types,
    _map_output_values,
    _normalize_model_info,
    _parse_openapi_schema,
    _parse_model_info_text,
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
