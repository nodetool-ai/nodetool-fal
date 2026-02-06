from __future__ import annotations

import hashlib
import json
import os
import re
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
    - Run custom endpoints by sharing model info (llms.txt)
    - Build flexible pipelines that depend on runtime model selection
    - Explore model inputs/outputs directly from OpenAPI metadata
    """

    _is_dynamic = True
    _supports_dynamic_outputs = True

    model_info: str = Field(
        default="",
        description=(
            "fal.ai llms.txt URL, fal.ai model URL, endpoint id, or raw llms.txt "
            "content used to derive the OpenAPI schema."
        ),
    )

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._prime_schema_outputs()

    @classmethod
    def get_basic_fields(cls):
        return ["model_info"]

    @classmethod
    def _cache_dir(cls) -> Path:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return CACHE_DIR

    def _prime_schema_outputs(self) -> None:
        model_info_text, model_info_url, endpoint_hint = _normalize_model_info(
            self.model_info
        )
        cache_key = _cache_key_for_model_info(
            model_info_text, model_info_url, endpoint_hint
        )
        if not cache_key:
            return

        cached = _load_cached_schema(self._cache_dir(), cache_key)
        if cached is not None:
            bundle = _parse_openapi_schema(
                cached["openapi"], cached.get("llm_info"), endpoint_hint=endpoint_hint
            )
            self._set_dynamic_outputs(bundle)
            _set_ui_schema_metadata(self._ui_properties, bundle, source="cache")

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        bundle = await self._load_schema_bundle(context)
        self._set_dynamic_outputs(bundle)

        input_values = dict(self.dynamic_properties)
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
        model_info_text, model_info_url, endpoint_hint = _normalize_model_info(
            self.model_info
        )
        if not model_info_text and not model_info_url:
            raise ValueError("model_info is required and cannot be empty")

        openapi_url: str | None = None
        endpoint_id: str | None = endpoint_hint
        llm_info = model_info_text

        cache_key = _cache_key_for_model_info(
            model_info_text, model_info_url, endpoint_hint
        )
        cached = _load_cached_schema(self._cache_dir(), cache_key)
        if cached is not None:
            bundle = _parse_openapi_schema(
                cached["openapi"], cached.get("llm_info"), endpoint_hint=endpoint_hint
            )
            _set_ui_schema_metadata(self._ui_properties, bundle, source="cache")
            return bundle

        if model_info_url:
            if _is_openapi_url(model_info_url):
                openapi_url = model_info_url
                endpoint_id = endpoint_id or _endpoint_from_openapi_url(model_info_url)
            else:
                llm_info = await _fetch_model_info(model_info_url)

        if llm_info:
            endpoint_id, openapi_url = _parse_model_info_text(llm_info, endpoint_id)

        if openapi_url is None and endpoint_id:
            openapi_url = _openapi_url_for_endpoint(endpoint_id)

        if openapi_url is None:
            raise ValueError("Unable to resolve an OpenAPI schema URL from model_info")

        openapi = await _fetch_openapi(openapi_url)
        _save_cached_schema(self._cache_dir(), cache_key, openapi, llm_info)
        bundle = _parse_openapi_schema(openapi, llm_info, endpoint_hint=endpoint_id)
        _set_ui_schema_metadata(self._ui_properties, bundle, source="remote")
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


async def _fetch_openapi(openapi_url: str) -> dict[str, Any]:
    timeout = httpx.Timeout(20.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        schema_resp = await client.get(openapi_url)
        schema_resp.raise_for_status()
        return schema_resp.json()


async def _fetch_model_info(model_info_url: str) -> str:
    timeout = httpx.Timeout(20.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(model_info_url)
        resp.raise_for_status()
        return resp.text


def _normalize_model_info(
    model_info: str,
) -> tuple[str | None, str | None, str | None]:
    normalized = model_info.strip()
    if not normalized:
        return None, None, None
    if _is_url(normalized):
        llms_url = _coerce_llms_url(normalized)
        endpoint_hint = _endpoint_from_llms_url(llms_url) if llms_url else None
        return None, llms_url or normalized, endpoint_hint
    if _looks_like_endpoint_id(normalized):
        llms_url = f"https://fal.ai/models/{normalized}/llms.txt"
        return None, llms_url, normalized
    return normalized, None, None


def _cache_key_for_model_info(
    model_info_text: str | None,
    model_info_url: str | None,
    endpoint_hint: str | None,
) -> str | None:
    if endpoint_hint:
        return endpoint_hint
    if model_info_url:
        return hashlib.sha256(model_info_url.encode("utf-8")).hexdigest()
    if model_info_text:
        return hashlib.sha256(model_info_text.encode("utf-8")).hexdigest()
    return None


def _is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _looks_like_endpoint_id(value: str) -> bool:
    return "/" in value and " " not in value and "\n" not in value


def _coerce_llms_url(value: str) -> str | None:
    if value.endswith("/llms.txt"):
        return value
    parsed = urlparse(value)
    if parsed.netloc.endswith("fal.ai") and parsed.path.startswith("/models/"):
        return f"{value.rstrip('/')}/llms.txt"
    return value


def _is_openapi_url(value: str) -> bool:
    return "openapi.json?endpoint_id=" in value


def _openapi_url_for_endpoint(endpoint_id: str) -> str:
    return f"https://fal.ai/api/openapi/queue/openapi.json?endpoint_id={endpoint_id}"


def _endpoint_from_llms_url(model_info_url: str | None) -> str | None:
    if not model_info_url:
        return None
    try:
        parsed = urlparse(model_info_url)
    except Exception:
        return None
    if not parsed.netloc.endswith("fal.ai"):
        return None
    if not parsed.path.startswith("/models/"):
        return None
    if not parsed.path.endswith("/llms.txt"):
        return None
    endpoint_path = parsed.path.replace("/models/", "").replace("/llms.txt", "")
    return endpoint_path.strip("/")


def _endpoint_from_openapi_url(openapi_url: str) -> str | None:
    parsed = urlparse(openapi_url)
    query = parse_qs(parsed.query)
    endpoint_candidates = query.get("endpoint_id") or []
    if endpoint_candidates:
        return endpoint_candidates[0]
    return None


def _parse_model_info_text(
    model_info: str, endpoint_hint: str | None
) -> tuple[str | None, str | None]:
    endpoint_id = endpoint_hint
    openapi_url = None

    openapi_match = re.search(
        r"https?://[^\s`]+openapi\.json\?endpoint_id=[^`\s]+", model_info
    )
    if openapi_match:
        openapi_url = openapi_match.group(0)

    model_id_match = re.search(r"Model ID\*\*:\s*`([^`]+)`", model_info)
    if model_id_match:
        endpoint_id = model_id_match.group(1)

    endpoint_match = re.search(r"Endpoint\*\*:\s*`https?://[^/]+/([^`]+)`", model_info)
    if endpoint_match:
        endpoint_id = endpoint_match.group(1)

    if not endpoint_id and openapi_url:
        endpoint_id = _endpoint_from_openapi_url(openapi_url)

    return endpoint_id, openapi_url


def _set_ui_schema_metadata(
    ui_properties: dict[str, Any],
    bundle: FalSchemaBundle,
    *,
    source: str,
) -> None:
    ui_properties["fal_schema"] = {"endpoint_id": bundle.endpoint_id, "source": source}
    ui_properties["fal_input_schema"] = bundle.input_schema
    ui_properties["fal_output_schema"] = bundle.output_schema


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
    openapi: dict[str, Any],
    llm_info: str | None,
    *,
    endpoint_hint: str | None = None,
) -> FalSchemaBundle:
    metadata = (openapi.get("info") or {}).get("x-fal-metadata") or {}
    endpoint_id = metadata.get("endpointId", "")

    input_schema = _extract_schema_from_paths(openapi, method="post")
    output_schema = _extract_output_schema(openapi)

    if not endpoint_id:
        endpoint_id = endpoint_hint or _fallback_endpoint_id(openapi)

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
