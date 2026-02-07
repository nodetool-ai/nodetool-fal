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

    def parse(self, openapi_schema: dict[str, Any]) -> NodeSpec:
        """
        Parse an OpenAPI schema into a node specification.
        
        Args:
            openapi_schema: OpenAPI schema dictionary
            
        Returns:
            NodeSpec with all information needed to generate a node
        """
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
        
        # Look for GET endpoint with /requests/{request_id}
        for path, methods in paths.items():
            if "/requests/{request_id}" in path:
                get = methods.get("get")
                if not get:
                    continue
                
                responses = get.get("responses", {})
                response_200 = responses.get("200", {})
                content = response_200.get("content", {}).get("application/json", {})
                output_schema_ref = content.get("schema", {})
                
                if output_schema_ref:
                    return self._resolve_ref(schema, output_schema_ref)
        
        return {}

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
            # Check for enum
            enum_ref = None
            if "enum" in prop:
                enum_name = self._generate_enum_name(name)
                enum_def = EnumDef(
                    name=enum_name,
                    values=[(self._to_enum_value(v), v) for v in prop["enum"]],
                    description=prop.get("description", "")
                )
                enums.append(enum_def)
                enum_ref = enum_name
            
            # Determine Python type (pass field name for better detection)
            python_type = self._json_type_to_python(prop, enum_ref, name)
            
            # Determine default value
            default_value = self._get_default_value(prop, python_type, name in required)
            
            fields.append(FieldDef(
                name=name,
                python_type=python_type,
                default_value=default_value,
                description=prop.get("description", ""),
                field_type=field_type,
                required=name in required,
                enum_ref=enum_ref
            ))
        
        return fields

    def _json_type_to_python(self, prop: dict[str, Any], enum_ref: Optional[str], prop_name: str = "") -> str:
        """Convert JSON schema type to Python type."""
        if enum_ref:
            return enum_ref
        
        json_type = prop.get("type", "string")
        
        if json_type == "string":
            # Check for image URL patterns
            desc_lower = prop.get("description", "").lower()
            title_lower = prop.get("title", "").lower()
            name_lower = prop_name.lower()
            
            if "image" in desc_lower or "image" in title_lower or ("image" in name_lower and "_url" in name_lower):
                return "ImageRef"
            elif "video" in desc_lower or "video" in title_lower or ("video" in name_lower and "_url" in name_lower):
                return "VideoRef"
            elif "audio" in desc_lower or "audio" in title_lower or ("audio" in name_lower and "_url" in name_lower):
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
            item_type = self._json_type_to_python(items, None, "")
            return f"list[{item_type}]"
        elif json_type == "object":
            return "dict"
        
        return "Any"

    def _get_default_value(self, prop: dict[str, Any], python_type: str, required: bool) -> str:
        """Get default value for a field."""
        if "default" in prop:
            default = prop["default"]
            if isinstance(default, str):
                # For enum types, use the enum value
                if python_type not in ["str", "ImageRef", "VideoRef", "AudioRef"]:
                    # This is likely an enum, find the matching enum value
                    return f'{python_type}.{self._to_enum_value(default)}'
                return f'"{default}"'
            elif isinstance(default, bool):
                return str(default)
            elif isinstance(default, (int, float)):
                return str(default)
        
        # Generate sensible defaults based on type
        if "ImageRef" in python_type:
            return "ImageRef()"
        elif "VideoRef" in python_type:
            return "VideoRef()"
        elif "AudioRef" in python_type:
            return "AudioRef()"
        elif python_type == "str":
            return '""'
        elif python_type == "int":
            return "-1" if "seed" in prop.get("description", "").lower() else "0"
        elif python_type == "float":
            return "0.0"
        elif python_type == "bool":
            return "False"
        elif python_type.startswith("list"):
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
        # Examples:
        # "16:9" -> "RATIO_16_9"
        # "square_hd" -> "SQUARE_HD"
        # "5" -> "DURATION_5"
        
        if ":" in value:
            value = value.replace(":", "_")
            return f"RATIO_{value}".upper()
        
        # Handle numeric values
        if value.isdigit():
            return f"VALUE_{value}"
        
        return value.replace("-", "_").upper()
