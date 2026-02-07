"""
Node code generator.

This module generates Python code for FAL nodes from node specifications.
"""

from typing import Optional
from codegen.schema_parser import NodeSpec, FieldDef, EnumDef


class NodeGenerator:
    """Generates Python code for FAL nodes."""

    def __init__(self):
        self._config_basic_fields = None
        self._field_renames = {}

    def generate(self, spec: NodeSpec, config: Optional[dict] = None) -> str:
        """
        Generate Python code for a node.
        
        Args:
            spec: Node specification
            config: Optional configuration to override/customize generation
            
        Returns:
            Generated Python code as string
        """
        config = config or {}
        
        # Apply config overrides
        spec = self._apply_config(spec, config)
        
        lines = []
        
        # Imports
        lines.extend(self._generate_imports(spec))
        lines.append("")
        lines.append("")
        
        # Enums
        for enum_def in spec.enums:
            lines.extend(self._generate_enum(enum_def))
            lines.append("")
            lines.append("")
        
        # Node class
        lines.extend(self._generate_class(spec))
        
        return "\n".join(lines)

    def _apply_config(self, spec: NodeSpec, config: dict) -> NodeSpec:
        """Apply configuration overrides to spec."""
        if "docstring" in config:
            spec.docstring = config["docstring"]
        if "tags" in config:
            spec.tags = config["tags"]
        if "use_cases" in config:
            spec.use_cases = config["use_cases"]
        if "class_name" in config:
            spec.class_name = config["class_name"]
        if "basic_fields" in config:
            self._config_basic_fields = config["basic_fields"]
        
        # Track field renames for API parameter mapping
        self._field_renames = {}
        
        # Enum overrides - apply first
        enum_rename_map = {}
        if "enum_overrides" in config:
            for enum_def in spec.enums:
                if enum_def.name in config["enum_overrides"]:
                    old_name = enum_def.name
                    new_name = config["enum_overrides"][old_name]
                    enum_rename_map[old_name] = new_name
                    enum_def.name = new_name
        
        # Enum value overrides
        if "enum_value_overrides" in config:
            for enum_def in spec.enums:
                # Check if there's a value override for this enum
                for orig_enum_name, value_map in config["enum_value_overrides"].items():
                    # Check if this is the renamed enum
                    if enum_rename_map.get(orig_enum_name) == enum_def.name or orig_enum_name == enum_def.name:
                        enum_def.values = [
                            (value_map.get(enum_name, enum_name), value)
                            for enum_name, value in enum_def.values
                        ]
                        break
        
        # Update enum references and default values in fields
        for field in spec.input_fields + spec.output_fields:
            if field.enum_ref:
                old_enum_ref = field.enum_ref
                # Apply enum rename
                if old_enum_ref in enum_rename_map:
                    new_enum_ref = enum_rename_map[old_enum_ref]
                    field.enum_ref = new_enum_ref
                    
                    # Update python_type
                    if "|" in field.python_type:
                        # Handle optional enums
                        parts = field.python_type.split(" | ")
                        parts[0] = new_enum_ref
                        field.python_type = " | ".join(parts)
                    else:
                        field.python_type = new_enum_ref
                    
                    # Update default value if it references the enum
                    if field.default_value.startswith(old_enum_ref + "."):
                        enum_value = field.default_value.split(".")[1]
                        # Check for enum value renames
                        if "enum_value_overrides" in config and old_enum_ref in config["enum_value_overrides"]:
                            value_map = config["enum_value_overrides"][old_enum_ref]
                            enum_value = value_map.get(enum_value, enum_value)
                        field.default_value = f"{new_enum_ref}.{enum_value}"
        
        # Field overrides
        if "field_overrides" in config:
            for field in spec.input_fields:
                if field.name in config["field_overrides"]:
                    override = config["field_overrides"][field.name]
                    
                    # Handle field rename
                    if "name" in override:
                        original_name = field.name
                        new_name = override["name"]
                        self._field_renames[new_name] = original_name
                        field.name = new_name
                    
                    if "python_type" in override:
                        new_type = override["python_type"]
                        field.python_type = new_type
                        # If the new type looks like an enum (capitalized, no brackets), set enum_ref
                        if new_type and new_type[0].isupper() and "[" not in new_type and " | " not in new_type:
                            field.enum_ref = new_type
                    if "default_value" in override:
                        field.default_value = override["default_value"]
                    if "description" in override:
                        field.description = override["description"]
        
        return spec

    def _generate_imports(self, spec: NodeSpec) -> list[str]:
        """Generate import statements."""
        imports = [
            "from pydantic import Field",
            "from typing import Any",
        ]
        
        # Add enum import if needed
        if spec.enums:
            imports.insert(0, "from enum import Enum")
        
        # Determine which asset types are needed
        asset_types = set()
        for field in spec.input_fields + spec.output_fields:
            if "ImageRef" in field.python_type:
                asset_types.add("ImageRef")
            elif "VideoRef" in field.python_type:
                asset_types.add("VideoRef")
            elif "AudioRef" in field.python_type:
                asset_types.add("AudioRef")
        
        # Add output type
        if "ImageRef" in spec.output_type:
            asset_types.add("ImageRef")
        elif "VideoRef" in spec.output_type:
            asset_types.add("VideoRef")
        elif "AudioRef" in spec.output_type:
            asset_types.add("AudioRef")
        
        if asset_types:
            imports.append(f"from nodetool.metadata.types import {', '.join(sorted(asset_types))}")
        
        imports.extend([
            "from nodetool.nodes.fal.fal_node import FALNode",
            "from nodetool.workflows.processing_context import ProcessingContext",
        ])
        
        return imports

    def _generate_enum(self, enum_def: EnumDef) -> list[str]:
        """Generate enum definition."""
        lines = [f"class {enum_def.name}(Enum):"]
        
        if enum_def.description:
            # Handle multi-line descriptions properly
            # Strip leading/trailing whitespace and ensure proper indentation
            desc_lines = enum_def.description.strip().split('\n')
            lines.append('    """')
            for desc_line in desc_lines:
                # Remove excessive leading whitespace but preserve some indentation
                stripped = desc_line.strip()
                if stripped:
                    lines.append(f'    {stripped}')
                # Skip completely empty lines in docstrings to avoid confusing the enum extractor
            lines.append('    """')
        
        for enum_name, value in enum_def.values:
            lines.append(f'    {enum_name} = "{value}"')
        
        return lines

    def _generate_class(self, spec: NodeSpec) -> list[str]:
        """Generate node class definition."""
        lines = [f"class {spec.class_name}(FALNode):"]
        
        # Docstring
        lines.extend(self._generate_docstring(spec))
        lines.append("")
        
        # Fields
        for field in spec.input_fields:
            lines.extend(self._generate_field(field))
        
        lines.append("")
        
        # Process method
        lines.extend(self._generate_process_method(spec))
        
        lines.append("")
        
        # get_basic_fields method
        lines.extend(self._generate_basic_fields_method(spec))
        
        return lines

    def _generate_docstring(self, spec: NodeSpec) -> list[str]:
        """Generate docstring for node class."""
        lines = ['    """']
        
        # Add description
        if spec.docstring:
            lines.append(f"    {spec.docstring}")
        else:
            lines.append(f"    {spec.class_name} node for {spec.endpoint_id}")
        
        # Add tags
        if spec.tags:
            lines.append(f"    {', '.join(spec.tags)}")
        else:
            lines.append("    fal, ai, generation")
        
        
        # Add use cases
        if spec.use_cases:
            lines.append("")
            lines.append("    Use cases:")
            for use_case in spec.use_cases:
                lines.append(f"    - {use_case}")
        
        lines.append('    """')
        
        return lines

    def _generate_field(self, field: FieldDef) -> list[str]:
        """Generate field definition."""
        lines = []
        
        # Determine Field parameters
        params = [f"default={field.default_value}"]
        
        if field.description:
            # Escape quotes and newlines in description
            desc = field.description.replace('"', '\\"')
            # Replace newlines and extra whitespace with single space
            desc = " ".join(desc.split())
            params.append(f'description="{desc}"')
        
        field_params = ", ".join(params)
        
        # Handle optional enums
        field_type = field.python_type
        if field.enum_ref and not field.required and field.default_value == "None":
            field_type = f"{field.python_type} | None"
        
        lines.append(f"    {field.name}: {field_type} = Field(")
        lines.append(f"        {field_params}")
        lines.append("    )")
        
        return lines

    def _generate_process_method(self, spec: NodeSpec) -> list[str]:
        """Generate async process method."""
        lines = [
            f"    async def process(self, context: ProcessingContext) -> {spec.output_type}:",
        ]
        
        # Convert input images/videos/audio to required format
        image_fields = []
        for field in spec.input_fields:
            if "ImageRef" in field.python_type:
                lines.append(f"        {field.name}_base64 = await context.image_to_base64(self.{field.name})")
                image_fields.append(field.name)
        
        # Build arguments dict
        lines.append("        arguments = {")
        
        for field in spec.input_fields:
            # Get the API parameter name (use original name if field was renamed)
            api_param_name = self._field_renames.get(field.name, field.name)
            
            if field.name in image_fields:
                # Use the API parameter name for image URLs
                lines.append(f'            "{api_param_name}": f"data:image/png;base64,{{{field.name}_base64}}",')
            elif field.enum_ref:
                # Handle optional enums
                if not field.required and field.default_value == "None":
                    lines.append(f'            "{api_param_name}": self.{field.name}.value if self.{field.name} else None,')
                else:
                    lines.append(f'            "{api_param_name}": self.{field.name}.value,')
            else:
                lines.append(f'            "{api_param_name}": self.{field.name},')
        
        lines.append("        }")
        lines.append("")
        
        # Filter out None values
        lines.append("        # Remove None values")
        lines.append("        arguments = {k: v for k, v in arguments.items() if v is not None}")
        lines.append("")
        
        # Submit request
        lines.append("        res = await self.submit_request(")
        lines.append("            context=context,")
        lines.append(f'            application="{spec.endpoint_id}",')
        lines.append("            arguments=arguments,")
        lines.append("        )")
        
        # Return output
        if spec.output_type == "VideoRef":
            lines.append('        assert "video" in res')
            lines.append('        return VideoRef(uri=res["video"]["url"])')
        elif spec.output_type == "ImageRef":
            lines.append('        assert "images" in res')
            lines.append('        assert len(res["images"]) > 0')
            lines.append('        return ImageRef(uri=res["images"][0]["url"])')
        elif spec.output_type == "AudioRef":
            lines.append('        assert "audio" in res')
            lines.append('        return AudioRef(uri=res["audio"]["url"])')
        else:
            lines.append("        return res")
        
        return lines

    def _generate_basic_fields_method(self, spec: NodeSpec) -> list[str]:
        """Generate get_basic_fields class method."""
        # Use config basic fields if available
        if self._config_basic_fields:
            basic_fields = self._config_basic_fields
        else:
            # Select up to 5 most important fields
            basic_fields = [f.name for f in spec.input_fields[:5]]
        
        fields_str = ", ".join(f'"{f}"' for f in basic_fields)
        
        return [
            "    @classmethod",
            "    def get_basic_fields(cls):",
            f"        return [{fields_str}]",
        ]
