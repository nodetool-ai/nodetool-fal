from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
from pydantic import Field

from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.metadata.types import (
    AssetRef,
    AudioRef,
    BaseType,
    DocumentRef,
    ImageRef,
    Model3DRef,
    VideoRef,
    asset_types,
)
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext

CACHE_DIR = Path(
    os.getenv(
        "NODETOOL_FAL_SCHEMA_CACHE",
        os.path.join(Path.home(), ".cache", "nodetool", "fal_schema"),
    )
)


@dataclass(frozen=True)
class FalSchemaBundle:
    endpoint_id: str
    openapi: dict[str, Any]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    llm_info: str | None


class DynamicFalSchema(FALNode):
    """
    Dynamic FAL schema-driven node for running any fal.ai endpoint.
    fal, schema, dynamic, openapi, inference, runtime, model

    Use cases:
    - Call new fal.ai endpoints without adding new Python nodes
    - Prototype workflows with experimental FAL models
    - Run custom endpoints by pasting their OpenAPI schema
    - Build flexible pipelines that depend on runtime model selection
    - Explore model inputs/outputs directly from OpenAPI metadata
    """

    _is_dynamic = True
    _supports_dynamic_outputs = True

    model_url: str = Field(
        default="",
        description="fal.ai model URL (e.g. https://fal.ai/models/fal-ai/flux/dev)",
    )
    endpoint_id: str = Field(
        default="",
        description="FAL endpoint id (e.g. fal-ai/flux/dev)",
    )
    schema_url: str = Field(
        default="",
        description="URL to the OpenAPI schema JSON for the endpoint",
    )
    openapi_json: dict[str, Any] | str | None = Field(
        default=None,
        description="Raw OpenAPI schema JSON (string or object)",
    )
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional input values for the endpoint (overrides dynamic properties)",
    )

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._prime_schema_outputs()

    @classmethod
    def get_basic_fields(cls):
        return ["model_url", "endpoint_id", "schema_url", "openapi_json"]

    @classmethod
    def _cache_dir(cls) -> Path:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return CACHE_DIR

    def _prime_schema_outputs(self) -> None:
        schema_payload = _parse_schema_payload(self.openapi_json)
        if schema_payload:
            bundle = _parse_openapi_schema(schema_payload, llm_info=None)
            self._set_dynamic_outputs(bundle)
            self._ui_properties["fal_schema"] = {
                "endpoint_id": bundle.endpoint_id,
                "source": "openapi_json",
            }
            return

        cache_key, _schema_url, _llm_url = _resolve_urls(
            model_url=self.model_url,
            endpoint_id=self.endpoint_id,
            schema_url=self.schema_url,
        )
        if not cache_key:
            return

        cached = _load_cached_schema(self._cache_dir(), cache_key)
        if cached is not None:
            bundle = _parse_openapi_schema(cached["openapi"], cached.get("llm_info"))
            self._set_dynamic_outputs(bundle)
            self._ui_properties["fal_schema"] = {
                "endpoint_id": bundle.endpoint_id,
                "source": "cache",
            }

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        bundle = await self._load_schema_bundle(context)
        self._set_dynamic_outputs(bundle)

        input_values = self.inputs if self.inputs else dict(self.dynamic_properties)
        arguments = await self._build_arguments(
            bundle.openapi,
            bundle.input_schema,
            input_values,
            context,
        )

        res = await self.submit_request(
            context=context,
            application=bundle.endpoint_id,
            arguments=arguments,
        )

        outputs = _map_output_values(bundle.openapi, bundle.output_schema, res)
        if not outputs:
            return {"result": res}
        return outputs

    def find_output_instance(self, name: str):
        slot = super().find_output_instance(name)
        if slot is not None:
            return slot
        if self._supports_dynamic_outputs:
            return _make_output_slot(name, TypeMetadata(type="any"))
        return None

    async def _load_schema_bundle(self, context: ProcessingContext) -> FalSchemaBundle:
        schema_payload = _parse_schema_payload(self.openapi_json)
        if schema_payload:
            bundle = _parse_openapi_schema(schema_payload, llm_info=None)
            self._ui_properties["fal_schema"] = {
                "endpoint_id": bundle.endpoint_id,
                "source": "openapi_json",
            }
            return bundle

        cache_key, schema_url, llm_url = _resolve_urls(
            model_url=self.model_url,
            endpoint_id=self.endpoint_id,
            schema_url=self.schema_url,
        )
        if not schema_url:
            raise ValueError(
                "A model_url, endpoint_id, schema_url, or openapi_json is required"
            )

        cached = _load_cached_schema(self._cache_dir(), cache_key)
        if cached is not None:
            bundle = _parse_openapi_schema(cached["openapi"], cached.get("llm_info"))
            self._ui_properties["fal_schema"] = {
                "endpoint_id": bundle.endpoint_id,
                "source": "cache",
            }
            return bundle

        openapi, llm_info = await _fetch_openapi_and_llm(schema_url, llm_url)
        _save_cached_schema(self._cache_dir(), cache_key, openapi, llm_info)
        bundle = _parse_openapi_schema(openapi, llm_info)
        self._ui_properties["fal_schema"] = {
            "endpoint_id": bundle.endpoint_id,
            "source": "remote",
        }
        if llm_info:
            self._ui_properties["fal_llm_info"] = llm_info
        return bundle

    def _set_dynamic_outputs(self, bundle: FalSchemaBundle) -> None:
        outputs = _build_output_types(bundle.openapi, bundle.output_schema)
        if not outputs:
            outputs = {"result": TypeMetadata(type="dict")}
        self._dynamic_outputs = outputs

    async def _build_arguments(
        self,
        openapi: dict[str, Any],
        input_schema: dict[str, Any],
        input_values: dict[str, Any],
        context: ProcessingContext,
    ) -> dict[str, Any]:
        if not isinstance(input_values, dict):
            raise ValueError("Input values must be a dictionary")

        schema_props = input_schema.get("properties", {})
        required_props = set(input_schema.get("required", []))

        arguments: dict[str, Any] = {}
        for name, prop_schema in schema_props.items():
            if name not in input_values:
                if name in required_props:
                    raise ValueError(f"Missing required input: {name}")
                continue
            value = input_values.get(name)
            if value is None:
                if name in required_props:
                    raise ValueError(f"Missing required input: {name}")
                continue
            arguments[name] = await _coerce_input_value(
                openapi,
                name,
                prop_schema,
                value,
                context,
                self,
            )

        return arguments


async def _fetch_openapi_and_llm(
    schema_url: str, llm_url: str | None
) -> tuple[dict[str, Any], str | None]:
    timeout = httpx.Timeout(20.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        schema_resp = await client.get(schema_url)
        schema_resp.raise_for_status()
        openapi = schema_resp.json()

        llm_info = None
        if llm_url:
            llm_resp = await client.get(llm_url)
            if llm_resp.status_code == 200:
                llm_info = llm_resp.text
        return openapi, llm_info


def _parse_schema_payload(
    openapi_json: dict[str, Any] | str | None,
) -> dict[str, Any] | None:
    if openapi_json is None:
        return None
    if isinstance(openapi_json, dict):
        return openapi_json
    if isinstance(openapi_json, str) and openapi_json.strip():
        return json.loads(openapi_json)
    return None


def _resolve_urls(
    *,
    model_url: str,
    endpoint_id: str,
    schema_url: str,
) -> tuple[str | None, str | None, str | None]:
    resolved_endpoint = endpoint_id.strip("/") if endpoint_id else ""
    resolved_schema_url = schema_url.strip()
    resolved_llm_url = None

    if model_url:
        candidate = _endpoint_from_model_url(model_url)
        if candidate:
            resolved_endpoint = candidate

    if resolved_endpoint:
        resolved_schema_url = (
            resolved_schema_url
            or f"https://fal.ai/api/openapi/queue/openapi.json?endpoint_id={resolved_endpoint}"
        )
        resolved_llm_url = f"https://fal.ai/models/{resolved_endpoint}/llms.txt"

    if not resolved_endpoint and resolved_schema_url:
        parsed = urlparse(resolved_schema_url)
        query = parse_qs(parsed.query)
        endpoint_candidates = query.get("endpoint_id") or []
        if endpoint_candidates:
            resolved_endpoint = endpoint_candidates[0]
            resolved_llm_url = f"https://fal.ai/models/{resolved_endpoint}/llms.txt"

    cache_key = None
    if resolved_endpoint:
        cache_key = resolved_endpoint
    elif resolved_schema_url:
        cache_key = hashlib.sha256(resolved_schema_url.encode("utf-8")).hexdigest()

    return cache_key, resolved_schema_url or None, resolved_llm_url


def _endpoint_from_model_url(model_url: str) -> str | None:
    try:
        parsed = urlparse(model_url)
    except Exception:
        return None
    if not parsed.netloc.endswith("fal.ai"):
        return None
    if not parsed.path.startswith("/models/"):
        return None
    return parsed.path.replace("/models/", "").strip("/")


def _load_cached_schema(
    cache_dir: Path, cache_key: str | None
) -> dict[str, Any] | None:
    if not cache_key:
        return None
    schema_path = cache_dir / f"{cache_key}.openapi.json"
    if not schema_path.exists():
        return None
    llm_path = cache_dir / f"{cache_key}.llms.txt"
    llm_info = None
    if llm_path.exists():
        llm_info = llm_path.read_text(encoding="utf-8")
    return {
        "openapi": json.loads(schema_path.read_text(encoding="utf-8")),
        "llm_info": llm_info,
    }


def _save_cached_schema(
    cache_dir: Path,
    cache_key: str | None,
    openapi: dict[str, Any],
    llm_info: str | None,
) -> None:
    if not cache_key:
        return
    schema_path = cache_dir / f"{cache_key}.openapi.json"
    schema_path.write_text(json.dumps(openapi), encoding="utf-8")
    if llm_info:
        llm_path = cache_dir / f"{cache_key}.llms.txt"
        llm_path.write_text(llm_info, encoding="utf-8")


def _parse_openapi_schema(
    openapi: dict[str, Any], llm_info: str | None
) -> FalSchemaBundle:
    metadata = (openapi.get("info") or {}).get("x-fal-metadata") or {}
    endpoint_id = metadata.get("endpointId", "")

    input_schema = _extract_schema_from_paths(openapi, method="post")
    output_schema = _extract_output_schema(openapi)

    if not endpoint_id:
        endpoint_id = _fallback_endpoint_id(openapi)

    if not input_schema:
        raise ValueError("Failed to locate input schema in OpenAPI document")
    if not output_schema:
        raise ValueError("Failed to locate output schema in OpenAPI document")

    return FalSchemaBundle(
        endpoint_id=endpoint_id,
        openapi=openapi,
        input_schema=input_schema,
        output_schema=output_schema,
        llm_info=llm_info,
    )


def _fallback_endpoint_id(openapi: dict[str, Any]) -> str:
    paths = openapi.get("paths", {})
    for path in paths:
        if path.startswith("/"):
            return path.strip("/")
    return ""


def _extract_schema_from_paths(openapi: dict[str, Any], method: str) -> dict[str, Any]:
    paths = openapi.get("paths", {})
    for path, methods in paths.items():
        entry = (methods or {}).get(method)
        if not entry:
            continue
        request_body = entry.get("requestBody")
        if not request_body:
            continue
        content = (request_body.get("content") or {}).get("application/json") or {}
        schema = content.get("schema")
        if schema:
            return _resolve_schema_ref(openapi, schema)
    return {}


def _extract_output_schema(openapi: dict[str, Any]) -> dict[str, Any]:
    paths = openapi.get("paths", {})
    candidate_schema = {}
    for path, methods in paths.items():
        entry = (methods or {}).get("get")
        if not entry:
            continue
        responses = entry.get("responses") or {}
        response = responses.get("200") or responses.get(200)
        if not response:
            continue
        content = (response.get("content") or {}).get("application/json") or {}
        schema = content.get("schema")
        if not schema:
            continue
        resolved = _resolve_schema_ref(openapi, schema)
        if path.endswith("/requests/{request_id}"):
            return resolved
        if not _is_queue_status_schema(resolved):
            candidate_schema = resolved
    return candidate_schema


def _is_queue_status_schema(schema: dict[str, Any]) -> bool:
    title = schema.get("title", "")
    if title.lower() == "queuestatus":
        return True
    properties = schema.get("properties", {})
    return "status" in properties and "request_id" in properties


def _resolve_schema_ref(
    openapi: dict[str, Any], schema: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {}
    if "$ref" in schema:
        return _resolve_ref(openapi, schema["$ref"])
    if "oneOf" in schema:
        options = schema.get("oneOf") or []
        if options:
            return _resolve_schema_ref(openapi, options[0])
    if "anyOf" in schema:
        options = schema.get("anyOf") or []
        if options:
            return _resolve_schema_ref(openapi, options[0])
    if "allOf" in schema:
        return _merge_all_of(openapi, schema["allOf"])
    return schema


def _merge_all_of(
    openapi: dict[str, Any], schemas: list[dict[str, Any]]
) -> dict[str, Any]:
    merged: dict[str, Any] = {"type": "object", "properties": {}}
    required: list[str] = []
    for entry in schemas:
        resolved = _resolve_schema_ref(openapi, entry)
        if not resolved:
            continue
        if "properties" in resolved:
            merged["properties"].update(resolved.get("properties") or {})
        if "required" in resolved:
            required.extend(resolved.get("required") or [])
        for key in ("title", "description", "type", "items"):
            if key in resolved and key not in merged:
                merged[key] = resolved[key]
    if required:
        merged["required"] = sorted(set(required))
    return merged


def _resolve_ref(openapi: dict[str, Any], ref: str) -> dict[str, Any]:
    if not ref.startswith("#/"):
        return {}
    parts = ref.lstrip("#/").split("/")
    current: Any = openapi
    for part in parts:
        if not isinstance(current, dict):
            return {}
        current = current.get(part)
        if current is None:
            return {}
    if isinstance(current, dict):
        return _resolve_schema_ref(openapi, current)
    return {}


async def _coerce_input_value(
    openapi: dict[str, Any],
    name: str,
    schema: dict[str, Any],
    value: Any,
    context: ProcessingContext,
    node: DynamicFalSchema,
) -> Any:
    resolved = _resolve_schema_ref(openapi, schema)

    asset_value = _coerce_asset_ref(value)
    if asset_value is not None:
        return await _serialize_asset_ref(asset_value, context, node)

    if resolved.get("type") == "array":
        item_schema = resolved.get("items") or {}
        if not isinstance(value, list):
            value = [value]
        return [
            await _coerce_input_value(openapi, name, item_schema, item, context, node)
            for item in value
            if item is not None
        ]

    if resolved.get("type") == "object" and isinstance(value, dict):
        properties = resolved.get("properties") or {}
        required = set(resolved.get("required", []))
        output: dict[str, Any] = {}
        for key, prop_schema in properties.items():
            if key not in value:
                if key in required:
                    raise ValueError(f"Missing required input: {name}.{key}")
                continue
            nested_value = value.get(key)
            if nested_value is None:
                if key in required:
                    raise ValueError(f"Missing required input: {name}.{key}")
                continue
            output[key] = await _coerce_input_value(
                openapi, key, prop_schema, nested_value, context, node
            )
        return output

    return value


def _coerce_asset_ref(value: Any) -> AssetRef | None:
    if isinstance(value, AssetRef):
        return value
    if isinstance(value, BaseType):
        if isinstance(value, AssetRef):
            return value
        return None
    if isinstance(value, dict) and value.get("type") in asset_types:
        try:
            return AssetRef.from_dict(value)
        except Exception:
            return None
    return None


async def _serialize_asset_ref(
    asset_ref: AssetRef,
    context: ProcessingContext,
    node: DynamicFalSchema,
) -> str:
    if asset_ref.uri and asset_ref.uri.startswith(("http://", "https://", "data:")):
        return asset_ref.uri

    if isinstance(asset_ref, ImageRef):
        image_base64 = await context.image_to_base64(asset_ref)
        return f"data:image/png;base64,{image_base64}"

    content_type = _asset_content_type(asset_ref)
    client = await node.get_client(context)
    asset_bytes = await context.asset_to_bytes(asset_ref)
    return await client.upload(asset_bytes, content_type)


def _asset_content_type(asset_ref: AssetRef) -> str:
    if isinstance(asset_ref, VideoRef):
        return "video/mp4"
    if isinstance(asset_ref, AudioRef):
        return "audio/mp3"
    if isinstance(asset_ref, DocumentRef):
        return "application/pdf"
    if isinstance(asset_ref, Model3DRef):
        return "model/gltf-binary"
    return "application/octet-stream"


def _map_output_values(
    openapi: dict[str, Any], output_schema: dict[str, Any], response: dict[str, Any]
) -> dict[str, Any]:
    properties = output_schema.get("properties") or {}
    output: dict[str, Any] = {}
    for name, schema in properties.items():
        if name not in response:
            continue
        output[name] = _map_output_value(openapi, name, schema, response.get(name))
    return output


def _map_output_value(
    openapi: dict[str, Any], name: str, schema: dict[str, Any], value: Any
) -> Any:
    resolved = _resolve_schema_ref(openapi, schema)
    if value is None:
        return None

    if resolved.get("type") == "array":
        item_schema = resolved.get("items") or {}
        if not isinstance(value, list):
            return []
        return [_map_output_value(openapi, name, item_schema, item) for item in value]

    if _is_file_schema(resolved):
        return _map_file_output(name, value)

    if resolved.get("type") == "object" and isinstance(value, dict):
        properties = resolved.get("properties") or {}
        return {
            key: _map_output_value(openapi, key, properties.get(key, {}), val)
            for key, val in value.items()
            if key in properties
        }

    return value


def _is_file_schema(schema: dict[str, Any]) -> bool:
    properties = schema.get("properties") or {}
    return "url" in properties


def _map_file_output(name: str, value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    url = value.get("url")
    if not url:
        return value

    asset_type = _infer_asset_type(name)
    if asset_type == "image":
        return ImageRef(uri=url)
    if asset_type == "video":
        return VideoRef(uri=url)
    if asset_type == "audio":
        return AudioRef(uri=url)
    if asset_type == "document":
        return DocumentRef(uri=url)
    if asset_type == "model_3d":
        return Model3DRef(uri=url)
    return AssetRef(uri=url)


def _infer_asset_type(name: str) -> str:
    lowered = name.lower()
    if "video" in lowered or "gif" in lowered:
        return "video"
    if "audio" in lowered or "voice" in lowered or "sound" in lowered:
        return "audio"
    if "image" in lowered or "mask" in lowered or "frame" in lowered:
        return "image"
    if "document" in lowered or "pdf" in lowered or "doc" in lowered:
        return "document"
    if "3d" in lowered or "mesh" in lowered or "gltf" in lowered or "glb" in lowered:
        return "model_3d"
    return "asset"


def _build_output_types(
    openapi: dict[str, Any], output_schema: dict[str, Any]
) -> dict[str, TypeMetadata]:
    properties = output_schema.get("properties") or {}
    output_types: dict[str, TypeMetadata] = {}
    for name, schema in properties.items():
        output_types[name] = _infer_output_type(openapi, name, schema)
    return output_types


def _infer_output_type(
    openapi: dict[str, Any], name: str, schema: dict[str, Any]
) -> TypeMetadata:
    resolved = _resolve_schema_ref(openapi, schema)
    if resolved.get("type") == "array":
        item_schema = resolved.get("items") or {}
        return TypeMetadata(
            type="list",
            type_args=[_infer_output_type(openapi, name, item_schema)],
        )
    if _is_file_schema(resolved):
        asset_type = _infer_asset_type(name)
        if asset_type == "asset":
            return TypeMetadata(type="any")
        return TypeMetadata(type=asset_type)
    if resolved.get("type") == "boolean":
        return TypeMetadata(type="bool")
    if resolved.get("type") == "integer":
        return TypeMetadata(type="int")
    if resolved.get("type") == "number":
        return TypeMetadata(type="float")
    if resolved.get("type") == "string":
        return TypeMetadata(type="str")
    if resolved.get("type") == "object":
        return TypeMetadata(type="dict")
    return TypeMetadata(type="any")


def _make_output_slot(name: str, type_metadata: TypeMetadata):
    from nodetool.metadata.types import OutputSlot

    return OutputSlot(type=type_metadata, name=name)
