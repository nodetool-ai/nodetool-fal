"""
OpenAPI schema parser for extracting node information.

This module parses OpenAPI schemas to extract input/output definitions,
enums, and other metadata needed for code generation.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class EnumDef:
    """Definition of an enum to generate."""
    name: str
    values: list[tuple[str, str]]  # (enum_name, value)
    description: str = ""


@dataclass
class FieldDef:
    """Definition of a field/property."""
    name: str
    python_type: str
    default_value: str
    description: str
    field_type: str  # 'input' or 'output'
    required: bool = False
    enum_ref: Optional[str] = None  # Reference to enum class name if applicable
    nested_asset_key: Optional[str] = None  # If set, wrap asset under this key (e.g., "video_url" -> {"video_url": asset})
    parent_field: Optional[str] = None  # If set, this field belongs to a nested structure with this parent field name


@dataclass
class NodeSpec:
    """Complete specification for generating a node."""
    endpoint_id: str
    class_name: str
    docstring: str
    tags: list[str]
    use_cases: list[str]
    input_fields: list[FieldDef]
    output_type: str
    output_fields: list[FieldDef]  # For complex outputs
    enums: list[EnumDef]


class SchemaParser:
    """Parses OpenAPI schemas into node specifications."""

    def __init__(self):
        self._root_schema: dict[str, Any] = {}

    def parse(self, openapi_schema: dict[str, Any]) -> NodeSpec:
        """
        Parse an OpenAPI schema into a node specification.
        
        Args:
            openapi_schema: OpenAPI schema dictionary
            
        Returns:
            NodeSpec with all information needed to generate a node
        """
        self._root_schema = openapi_schema

        # Extract endpoint ID
        endpoint_id = self._extract_endpoint_id(openapi_schema)
        
        # Extract input schema
        input_schema = self._extract_input_schema(openapi_schema)
        
        # Extract output schema
        output_schema = self._extract_output_schema(openapi_schema)
        
        # Parse schemas into fields and enums
        enums = []
        input_fields = self._parse_properties(
            input_schema.get("properties", {}),
            input_schema.get("required", []),
            "input",
            enums
        )
        
        output_fields = self._parse_properties(
            output_schema.get("properties", {}),
            output_schema.get("required", []),
            "output",
            enums
        )
        
        # Determine output type
        output_type = self._determine_output_type(output_schema, output_fields)
        
        # Generate class name from endpoint
        class_name = self._generate_class_name(endpoint_id)
        
        return NodeSpec(
            endpoint_id=endpoint_id,
            class_name=class_name,
            docstring="",  # Will be filled by config
            tags=[],
            use_cases=[],
            input_fields=input_fields,
            output_type=output_type,
            output_fields=output_fields,
            enums=enums
        )

    def _extract_endpoint_id(self, schema: dict[str, Any]) -> str:
        """Extract endpoint ID from schema."""
        try:
            return schema["info"]["x-fal-metadata"]["endpointId"]
        except (KeyError, TypeError):
            # Fallback: try to extract from paths
            paths = schema.get("paths", {})
            for path in paths:
                if path.startswith("/"):
                    return path.strip("/")
            return ""

    def _extract_input_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Extract input schema from OpenAPI paths."""
        paths = schema.get("paths", {})
        for path, methods in paths.items():
            post = methods.get("post")
            if not post:
                continue
            
            request_body = post.get("requestBody", {})
            content = request_body.get("content", {}).get("application/json", {})
            input_schema_ref = content.get("schema", {})
            
            if input_schema_ref:
                return self._resolve_ref(schema, input_schema_ref)
        
        return {}

    def _extract_output_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Extract output schema from OpenAPI paths."""
        paths = schema.get("paths", {})
        candidate_schema = {}
        
        # Look for GET endpoints
        for path, methods in paths.items():
            get = methods.get("get")
            if not get:
                continue
            
            responses = get.get("responses", {})
            response_200 = responses.get("200") or responses.get(200)
            if not response_200:
                continue
                
            content = response_200.get("content", {}).get("application/json", {})
            output_schema_ref = content.get("schema")
            if not output_schema_ref:
                continue
            
            resolved = self._resolve_ref(schema, output_schema_ref)
            
            # Prefer the /requests/{request_id} path (actual result, not queue status)
            if path.endswith("/requests/{request_id}"):
                return resolved
            
            # Otherwise save as candidate if it's not a queue status schema
            if not self._is_queue_status_schema(resolved):
                candidate_schema = resolved
        
        return candidate_schema
    
    def _is_queue_status_schema(self, schema: dict[str, Any]) -> bool:
        """Check if a schema is a queue status schema (not the actual output)."""
        title = schema.get("title", "")
        if title.lower() == "queuestatus":
            return True
        properties = schema.get("properties", {})
        return "status" in properties and "request_id" in properties

    def _resolve_ref(self, schema: dict[str, Any], schema_obj: dict[str, Any]) -> dict[str, Any]:
        """Resolve $ref references in schema."""
        if not isinstance(schema_obj, dict):
            return {}
        
        if "$ref" in schema_obj:
            ref_path = schema_obj["$ref"]
            if ref_path.startswith("#/"):
                parts = ref_path.lstrip("#/").split("/")
                current = schema
                for part in parts:
                    current = current.get(part, {})
                return self._resolve_ref(schema, current)
        
        # Handle allOf
        if "allOf" in schema_obj:
            merged = {"type": "object", "properties": {}, "required": []}
            for sub_schema in schema_obj["allOf"]:
                resolved = self._resolve_ref(schema, sub_schema)
                if "properties" in resolved:
                    merged["properties"].update(resolved["properties"])
                if "required" in resolved:
                    merged["required"].extend(resolved["required"])
            return merged
        
        return schema_obj

    def _parse_properties(
        self,
        properties: dict[str, Any],
        required: list[str],
        field_type: str,
        enums: list[EnumDef]
    ) -> list[FieldDef]:
        """Parse properties into field definitions."""
        fields = []
        
        for name, prop in properties.items():
            normalized_prop, is_nullable = self._normalize_property_schema(prop)

            # Check if this is a nested asset structure (e.g., VideoConditioningInput)
            nested_asset_key, extra_fields = self._get_nested_asset_info(normalized_prop)
            
            if nested_asset_key:
                # Determine asset type based on the nested key
                if "video" in nested_asset_key.lower():
                    python_type = "VideoRef"
                elif "image" in nested_asset_key.lower():
                    python_type = "ImageRef"
                elif "audio" in nested_asset_key.lower():
                    python_type = "AudioRef"
                else:
                    python_type = "str"
                
                default_value = self._get_default_value(
                    {"type": "asset"},
                    python_type,
                    name in required,
                    nullable=is_nullable,
                )
                
                fields.append(FieldDef(
                    name=name,
                    python_type=python_type,
                    default_value=default_value,
                    description=prop.get("description", ""),
                    field_type=field_type,
                    required=name in required,
                    enum_ref=None,
                    nested_asset_key=nested_asset_key
                ))
                
                # Add extra fields from the nested schema (like start_frame_num)
                for extra in extra_fields:
                    fields.append(FieldDef(
                        name=extra["name"],
                        python_type=extra["python_type"],
                        default_value=extra["default_value"],
                        description=extra["description"],
                        field_type=field_type,
                        required=False,
                        enum_ref=None,
                        parent_field=name  # Link to the parent asset field
                    ))
                continue
            
            # Check for enum
            enum_ref = None
            enum_name = None
            if "enum" in normalized_prop:
                enum_name = self._generate_enum_name(name)
                enum_def = EnumDef(
                    name=enum_name,
                    values=[(self._to_enum_value(v), v) for v in normalized_prop["enum"]],
                    description=normalized_prop.get("description", "")
                )
                enums.append(enum_def)
                enum_ref = enum_name
            
            # Determine Python type (pass field name for better detection)
            python_type = self._json_type_to_python(normalized_prop, enum_ref, name)
            if is_nullable and "| None" not in python_type:
                python_type = f"{python_type} | None"
            
            # Determine default value (pass enum_name for proper default generation)
            default_value = self._get_default_value(
                normalized_prop,
                python_type,
                name in required,
                enum_name,
                nullable=is_nullable,
            )
            
            fields.append(FieldDef(
                name=name,
                python_type=python_type,
                default_value=default_value,
                description=normalized_prop.get("description", ""),
                field_type=field_type,
                required=name in required,
                enum_ref=enum_ref
            ))
        
        return fields

    def _normalize_property_schema(self, prop: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Normalize nullable unions (anyOf/oneOf) to a primary schema + nullable flag."""
        resolved = self._resolve_ref(self._root_schema, prop)
        if not isinstance(resolved, dict):
            return {}, False

        union_key = "anyOf" if "anyOf" in resolved else "oneOf" if "oneOf" in resolved else None
        is_nullable = bool(resolved.get("nullable", False))

        if union_key:
            variants = resolved.get(union_key, [])
            primary: dict[str, Any] | None = None
            for variant in variants:
                variant_resolved = self._resolve_ref(self._root_schema, variant)
                if not isinstance(variant_resolved, dict):
                    continue
                if variant_resolved.get("type") == "null":
                    is_nullable = True
                    continue
                if primary is None:
                    primary = variant_resolved
            normalized = dict(primary or {})
        else:
            normalized = dict(resolved)

        # Preserve wrapper-level metadata such as description/default/title.
        for key in ("title", "description", "default", "examples"):
            if key in resolved:
                normalized[key] = resolved[key]

        return normalized, is_nullable

    def _json_type_to_python(self, prop: dict[str, Any], enum_ref: Optional[str], prop_name: str = "") -> str:
        """Convert JSON schema type to Python type."""
        if enum_ref:
            return enum_ref
        
        json_type = prop.get("type", "string")
        
        if json_type == "string":
            # Check for image URL patterns - be more selective
            # Only treat as asset refs if field name ends with _url or is exactly image/video/audio
            name_lower = prop_name.lower()
            
            if name_lower.endswith("_url") or name_lower.endswith("_urls"):
                desc_lower = prop.get("description", "").lower()
                title_lower = prop.get("title", "").lower()
                
                if "image" in name_lower or "image" in desc_lower or "image" in title_lower:
                    return "ImageRef"
                elif "video" in name_lower or "video" in desc_lower or "video" in title_lower:
                    return "VideoRef"
                elif "audio" in name_lower or "audio" in desc_lower or "audio" in title_lower:
                    return "AudioRef"
            elif name_lower in ["image", "mask"]:
                # Specific known image input fields
                return "ImageRef"
            elif name_lower in ["video"]:
                return "VideoRef"
            elif name_lower in ["audio"]:
                return "AudioRef"
            
            return "str"
        elif json_type == "integer":
            return "int"
        elif json_type == "number":
            return "float"
        elif json_type == "boolean":
            return "bool"
        elif json_type == "array":
            items = prop.get("items", {})
            # Handle $ref in array items (complex object types)
            if "$ref" in items:
                ref_type = self._resolve_ref_type_name(items["$ref"])
                if ref_type:
                    return f"list[{ref_type}]"
            item_type = self._json_type_to_python(items, None, "")
            return f"list[{item_type}]"
        elif json_type == "object":
            return "dict"
        
        return "Any"

    def _resolve_ref_type_name(self, ref_path: str) -> Optional[str]:
        """Resolve a $ref path to the referenced schema's title or name.
        
        Returns the schema title (e.g., 'KlingV3MultiPromptElement') which
        corresponds to a BaseType subclass name.
        """
        if not ref_path.startswith("#/"):
            return None
        parts = ref_path.lstrip("#/").split("/")
        current = self._root_schema
        for part in parts:
            current = current.get(part, {})
            if not current:
                return None
        # Use the title from the resolved schema
        return current.get("title")

    def _get_nested_asset_info(self, prop: dict[str, Any]) -> tuple[Optional[str], list[dict[str, Any]]]:
        """Check if a property references a schema containing an asset URL field.
        
        Returns:
            Tuple of (nested_asset_key, extra_fields) where:
            - nested_asset_key: The key for the asset URL inside the nested object (e.g., "video_url")
            - extra_fields: List of additional field definitions from the nested schema
        """
        # Resolve the property schema to get its full structure
        resolved = self._resolve_ref(self._root_schema, prop)
        
        if not resolved or "properties" not in resolved:
            return None, []
        
        properties = resolved.get("properties", {})
        nested_asset_key = None
        asset_type = None
        extra_fields = []
        
        # Look for asset URL fields in the nested properties
        for key, sub_prop in properties.items():
            key_lower = key.lower()
            if key_lower.endswith("_url"):
                if "video" in key_lower:
                    nested_asset_key = key
                    asset_type = "VideoRef"
                elif "image" in key_lower:
                    nested_asset_key = key
                    asset_type = "ImageRef"
                elif "audio" in key_lower:
                    nested_asset_key = key
                    asset_type = "AudioRef"
            else:
                # Collect non-asset fields to add as separate node fields
                sub_type = sub_prop.get("type", "string")
                if sub_type == "integer":
                    python_type = "int"
                    default = "0"
                elif sub_type == "number":
                    python_type = "float"
                    default = "0.0"
                elif sub_type == "boolean":
                    python_type = "bool"
                    default = "False"
                else:
                    python_type = "str"
                    default = '""'
                
                extra_fields.append({
                    "name": key,
                    "python_type": python_type,
                    "default_value": default,
                    "description": sub_prop.get("description", ""),
                })
        
        return nested_asset_key, extra_fields

    def _get_default_value(
        self,
        prop: dict[str, Any],
        python_type: str,
        required: bool,
        enum_name: Optional[str] = None,
        nullable: bool = False,
    ) -> str:
        """Get default value for a field."""
        base_python_type = python_type.split(" | ")[0].strip()

        # Asset refs should always default to empty refs in nodetool nodes.
        if "ImageRef" in base_python_type:
            return "ImageRef()"
        if "VideoRef" in base_python_type:
            return "VideoRef()"
        if "AudioRef" in base_python_type:
            return "AudioRef()"

        if "default" in prop:
            default = prop["default"]
            if default is None:
                return "None"
            if isinstance(default, str):
                # For enum types, use the enum value
                if enum_name:
                    # Use enum name with correct value
                    enum_value = self._to_enum_value(default)
                    return f'{enum_name}.{enum_value}'
                elif base_python_type not in ["str", "ImageRef", "VideoRef", "AudioRef"]:
                    # This is likely an enum, find the matching enum value
                    enum_value = self._to_enum_value(default)
                    return f'{base_python_type}.{enum_value}'
                return f'"{default}"'
            elif isinstance(default, bool):
                return str(default)
            elif isinstance(default, (int, float)):
                return str(default)
        
        if nullable:
            return "None"
        
        # Generate sensible defaults based on type
        if base_python_type == "str":
            return '""'
        elif base_python_type == "int":
            return "-1" if "seed" in prop.get("description", "").lower() else "0"
        elif base_python_type == "float":
            return "0.0"
        elif base_python_type == "bool":
            return "False"
        elif base_python_type.startswith("list"):
            return "[]"
        elif required:
            return '""'
        
        return "None"

    def _determine_output_type(self, output_schema: dict[str, Any], output_fields: list[FieldDef]) -> str:
        """Determine the output type for the node."""
        properties = output_schema.get("properties", {})
        
        # Single output patterns
        if len(properties) == 1:
            prop_name = list(properties.keys())[0]
            if "video" in prop_name.lower():
                return "VideoRef"
            elif "image" in prop_name.lower():
                return "ImageRef"
            elif "audio" in prop_name.lower():
                return "AudioRef"
        
        # Check if video is present (even with other properties)
        if "video" in properties:
            return "VideoRef"
        
        # Check if images is present
        if "images" in properties:
            return "ImageRef"
        
        # Check if audio is present
        if "audio" in properties:
            return "AudioRef"
        
        # Multiple outputs or dict
        if len(properties) > 1:
            return "dict[str, Any]"
        
        return "Any"

    def _generate_class_name(self, endpoint_id: str) -> str:
        """Generate a class name from endpoint ID."""
        # Examples:
        # fal-ai/flux/dev -> FluxDev
        # fal-ai/luma-dream-machine/image-to-video -> LumaDreamMachineImageToVideo
        
        parts = endpoint_id.split("/")
        # Skip 'fal-ai' prefix
        if parts and parts[0] == "fal-ai":
            parts = parts[1:]
        
        # Convert kebab-case to PascalCase, also handle dots
        name_parts = []
        for part in parts:
            # Replace dots with nothing (v5.6 -> v56)
            part = part.replace(".", "")
            words = part.split("-")
            name_parts.extend([w.capitalize() for w in words])
        
        return "".join(name_parts)

    def _generate_enum_name(self, field_name: str) -> str:
        """Generate an enum class name from field name."""
        # Convert snake_case or kebab-case to PascalCase
        words = field_name.replace("-", "_").split("_")
        return "".join(w.capitalize() for w in words)

    def _to_enum_value(self, value: str) -> str:
        """Convert a string value to a valid Python enum name."""
        import re as _re
        # Examples:
        # "16:9" -> "RATIO_16_9"
        # "square_hd" -> "SQUARE_HD"
        # "5" -> "DURATION_5"
        # "3D Model" -> "MODEL_3D"
        # "Digital Art" -> "DIGITAL_ART"
        # "(No style)" -> "NO_STYLE"
        # "realistic_image/b_and_w" -> "REALISTIC_IMAGE__B_AND_W"
        # "X264 (.mp4)" -> "X264__MP4"
        # "DPM++ 2M" -> "DPM_PLUS_PLUS_2M"
        # "Who's Arrested?" -> "WHOS_ARRESTED"
        
        # Handle ratios early (before removing colons)
        if ":" in value and _re.match(r'^\d+:\d+$', value.strip()):
            value = value.replace(":", "_")
            return f"RATIO_{value}".upper()
        
        # Handle numeric values
        if value.strip().isdigit():
            return f"VALUE_{value.strip()}"
        
        # Replace ++ with _PLUS_PLUS, + with _PLUS
        value = value.replace("++", "_PLUS_PLUS_").replace("+", "_PLUS_")
        
        # Remove/replace special characters
        value = value.replace("(", "").replace(")", "").replace(",", "_")
        value = value.replace("'", "").replace("'", "").replace("\"", "")
        value = value.replace("!", "").replace("?", "").replace("&", "_AND_")
        value = value.replace(":", "_").replace(";", "_").replace("#", "_")
        value = value.replace("@", "_AT_").replace("$", "_")
        value = value.replace("~", "_").replace("`", "").replace("^", "_")
        value = value.replace("{", "").replace("}", "").replace("[", "").replace("]", "")
        value = value.replace("\\", "_").replace("|", "_").replace("=", "_")
        value = value.replace("<", "_").replace(">", "_")
        
        # Replace spaces, hyphens, slashes, dots with underscores
        # Use double underscore for slashes to make them stand out
        value = value.replace("/", "__").replace(" ", "_").replace("-", "_").replace(".", "_")
        
        # Convert to uppercase
        result = value.upper()
        
        # Collapse multiple underscores and strip leading/trailing underscores
        result = _re.sub(r'_+', '_', result).strip('_')
        
        # If starts with a digit, prefix with an appropriate word
        if result and result[0].isdigit():
            # Try to extract meaningful prefix from the rest of the string
            if "D" in result and result.index("D") < 3:
                # Like "3D" -> move to end: "MODEL_3D" or "ART_3D"
                parts = result.split("_")
                if len(parts) > 1:
                    # Move first part to end
                    result = "_".join(parts[1:] + [parts[0]])
                else:
                    result = f"VALUE_{result}"
            else:
                result = f"VALUE_{result}"
        
        # Final safety: ensure result is a valid identifier
        if not result or not result.isidentifier():
            # Replace any remaining invalid characters
            result = _re.sub(r'[^A-Za-z0-9_]', '_', result)
            result = _re.sub(r'_+', '_', result).strip('_')
            if not result:
                result = "VALUE_UNKNOWN"
            if result[0].isdigit():
                result = f"VALUE_{result}"
        
        return result
