"""
Node code generator.

This module generates Python code for FAL nodes from node specifications.
"""

from typing import Optional
from codegen.schema_parser import NodeSpec, FieldDef, EnumDef


class NodeGenerator:
    """Generates Python code for FAL nodes."""

    # Known BaseType subclass names that map to specific types
    KNOWN_BASE_TYPES = {
        "KlingV3MultiPromptElement",
        "KlingV3ImageElementInput",
        "KlingV3ComboElementInput",
        "LoraWeight",
        "LoRAWeight",
        "LoRAInput",
        "ControlLoraWeight",
        "ChronoLoraWeight",
        "EasyControlWeight",
        "ControlNet",
        "ControlNetUnion",
        "ControlNetUnionInput",
        "IPAdapter",
        "Embedding",
        "RGBColor",
        "PointPrompt",
        "PointPromptBase",
        "BoxPrompt",
        "BoxPromptBase",
        "BBoxPromptBase",
        "ElementInput",
        "OmniVideoElementInput",
        "DynamicMask",
        "Frame",
        "KeyframeTransition",
        "ImageCondition",
        "VideoCondition",
        "ImageConditioningInput",
        "VideoConditioningInput",
        "Track",
        "AudioTimeSpan",
        "InpaintSection",
        "DialogueBlock",
        "PronunciationDictionaryLocator",
        "Speaker",
        "Turn",
        "VibeVoiceSpeaker",
        "GuidanceInput",
        "ReferenceFace",
        "MoondreamInputParam",
        "ImageInput",
        "ReferenceImageInput",
        "SemanticImageInput",
    }

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
        
        # Prefix enum names with class name to avoid conflicts
        enum_name_map = {}
        for enum_def in spec.enums:
            prefixed_name = f"{spec.class_name}{enum_def.name}"
            enum_name_map[enum_def.name] = prefixed_name
            enum_def.name = prefixed_name
        
        # Update enum references in fields
        for field in spec.input_fields + spec.output_fields:
            if field.enum_ref and field.enum_ref in enum_name_map:
                field.enum_ref = enum_name_map[field.enum_ref]
                # Update python_type if it references the enum
                if field.python_type in enum_name_map:
                    field.python_type = enum_name_map[field.python_type]
                elif " | " in field.python_type:
                    parts = field.python_type.split(" | ")
                    new_parts = [enum_name_map.get(p, p) for p in parts]
                    field.python_type = " | ".join(new_parts)
                # Update default_value if it references the enum
                for old_name, new_name in enum_name_map.items():
                    if field.default_value.startswith(old_name + "."):
                        field.default_value = field.default_value.replace(old_name + ".", new_name + ".", 1)
        
        lines = []
        
        # Imports
        lines.extend(self._generate_imports(spec))
        lines.append("")
        lines.append("")
        
        # Enums at module level with prefixed names
        if spec.enums:
            for enum_def in spec.enums:
                lines.extend(self._generate_enum(enum_def, indent=0))
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

        # Special handling: normalize asset URL fields to nodetool-native names.
        self._normalize_image_urls_fields(spec)
        self._normalize_asset_url_fields(spec)
        self._normalize_asset_urls_fields(spec)
        
        return spec

    def _normalize_image_urls_fields(self, spec: NodeSpec) -> None:
        """Normalize `image_urls` input fields to `images: list[ImageRef]`."""
        for field in spec.input_fields:
            api_param_name = self._field_renames.get(field.name, field.name)
            if api_param_name != "image_urls":
                continue

            # Keep API mapping to image_urls while exposing a nodetool-native images field.
            if field.name in self._field_renames:
                del self._field_renames[field.name]
            field.name = "images"
            self._field_renames[field.name] = "image_urls"

            field.python_type = "list[ImageRef]"
            field.enum_ref = None
            if field.default_value == "None":
                field.default_value = "[]"

    def _normalize_asset_url_fields(self, spec: NodeSpec) -> None:
        """Normalize `*_url` fields to `*` for asset refs while preserving API names."""
        for field in spec.input_fields:
            api_param_name = self._field_renames.get(field.name, field.name)
            if not api_param_name.endswith("_url"):
                continue
            if not any(
                ref_type in field.python_type
                for ref_type in ("ImageRef", "AudioRef", "VideoRef")
            ):
                continue

            normalized_name = api_param_name.removesuffix("_url")
            if any(other.name == normalized_name and other is not field for other in spec.input_fields):
                continue

            # Preserve API mapping while exposing nodetool-native field names.
            if field.name in self._field_renames:
                del self._field_renames[field.name]
            field.name = normalized_name
            self._field_renames[field.name] = api_param_name

    def _normalize_asset_urls_fields(self, spec: NodeSpec) -> None:
        """Normalize asset list URL fields (e.g. `input_image_urls` -> `input_images`)."""
        for field in spec.input_fields:
            api_param_name = self._field_renames.get(field.name, field.name)
            if api_param_name == "image_urls":
                continue
            if not api_param_name.endswith("_urls"):
                continue
            if not any(asset in api_param_name for asset in ("image", "audio", "video")):
                continue

            normalized_name = f"{api_param_name.removesuffix('_urls')}s"
            if any(other.name == normalized_name and other is not field for other in spec.input_fields):
                continue

            if field.name in self._field_renames:
                del self._field_renames[field.name]
            field.name = normalized_name
            self._field_renames[field.name] = api_param_name

    def _generate_imports(self, spec: NodeSpec) -> list[str]:
        """Generate import statements."""
        imports = [
            "from pydantic import Field",
            "from typing import Any",
        ]
        
        # Add enum import if needed
        if spec.enums:
            imports.insert(0, "from enum import Enum")
        
        # Determine which asset types and BaseType subclasses are needed
        asset_types = set()
        base_type_classes = set()
        needs_base_type = False
        
        for field in spec.input_fields + spec.output_fields:
            if "ImageRef" in field.python_type:
                asset_types.add("ImageRef")
            elif "VideoRef" in field.python_type:
                asset_types.add("VideoRef")
            elif "AudioRef" in field.python_type:
                asset_types.add("AudioRef")
            # Detect BaseType subclass references (e.g., list[KlingV3MultiPromptElement])
            for bt_name in self.KNOWN_BASE_TYPES:
                if bt_name in field.python_type:
                    base_type_classes.add(bt_name)
                    needs_base_type = True
        
        # Add output type
        if "ImageRef" in spec.output_type:
            asset_types.add("ImageRef")
        elif "VideoRef" in spec.output_type:
            asset_types.add("VideoRef")
        elif "AudioRef" in spec.output_type:
            asset_types.add("AudioRef")
        
        # Build nodetool.metadata.types import
        metadata_types = sorted(asset_types)
        if needs_base_type:
            metadata_types = ["BaseType"] + metadata_types
        if metadata_types:
            imports.append(f"from nodetool.metadata.types import {', '.join(metadata_types)}")
        
        imports.extend([
            "from nodetool.nodes.fal.fal_node import FALNode",
            "from nodetool.workflows.processing_context import ProcessingContext",
        ])
        
        return imports

    def _generate_enum(self, enum_def: EnumDef, indent: int = 1) -> list[str]:
        """Generate enum definition as nested class.
        
        Args:
            enum_def: Enum definition
            indent: Indentation level (1 for nested in class, 0 for module level)
        """
        ind = "    " * indent
        lines = [f"{ind}class {enum_def.name}(str, Enum):"]
        
        if enum_def.description:
            # Handle multi-line descriptions properly
            # Strip leading/trailing whitespace from each line for clean output
            desc_lines = enum_def.description.strip().split('\n')
            lines.append(f'{ind}    """')
            for desc_line in desc_lines:
                # Remove all leading whitespace and add consistent indentation
                stripped = desc_line.strip()
                if stripped:
                    lines.append(f'{ind}    {stripped}')
                # Skip completely empty lines in docstrings to avoid confusing the enum extractor
            lines.append(f'{ind}    """')
        
        for enum_name, value in enum_def.values:
            lines.append(f'{ind}    {enum_name} = "{value}"')
        
        return lines

    def _generate_class(self, spec: NodeSpec) -> list[str]:
        """Generate node class definition."""
        lines = [f"class {spec.class_name}(FALNode):"]
        
        # Docstring
        lines.extend(self._generate_docstring(spec))
        lines.append("")
        
        # Note: Enums are now generated at module level, not nested
        
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

    def _is_base_type_list(self, field: FieldDef) -> bool:
        """Check if a field is a list of BaseType subclass instances."""
        for bt_name in self.KNOWN_BASE_TYPES:
            if bt_name in field.python_type:
                return True
        return False

    def _generate_process_method(self, spec: NodeSpec) -> list[str]:
        """Generate async process method."""
        lines = [
            f"    async def process(self, context: ProcessingContext) -> {spec.output_type}:",
        ]
        
        # Convert input images/videos/audio to required format
        image_fields = []
        image_list_fields = []
        video_fields = []
        video_list_fields = []
        nested_asset_fields = {}  # Maps field name to (nested_key, extra_field_names)
        
        # First pass: identify nested asset fields and their associated extra fields
        for field in spec.input_fields:
            if field.nested_asset_key:
                # Collect extra fields that belong to this nested structure
                # These are fields that have this field as their parent
                extra_fields = [
                    f.name for f in spec.input_fields
                    if f.parent_field == field.name
                ]
                nested_asset_fields[field.name] = (field.nested_asset_key, extra_fields)
        
        # Check if we need to upload videos
        needs_video_upload = any(
            field.python_type in ("VideoRef", "VideoRef | None")
            for field in spec.input_fields
        ) or any(
            field.python_type == "list[VideoRef]"
            for field in spec.input_fields
        )
        
        if needs_video_upload:
            lines.append("        client = await self.get_client(context)")
        
        for field in spec.input_fields:
            if field.python_type == "list[ImageRef]":
                lines.append(f"        {field.name}_data_urls = []")
                lines.append(f"        for image in self.{field.name} or []:")
                lines.append("            if image.is_empty():")
                lines.append("                continue")
                lines.append("            image_base64 = await context.image_to_base64(image)")
                lines.append(f'            {field.name}_data_urls.append(f"data:image/png;base64,{{image_base64}}")')
                image_list_fields.append(field.name)
            elif field.python_type == "list[VideoRef]":
                lines.append(f"        {field.name}_urls = []")
                lines.append(f"        for video in self.{field.name} or []:")
                lines.append("            if video.is_empty():")
                lines.append("                continue")
                lines.append(f"            video_bytes = await context.asset_to_bytes(video)")
                lines.append(f'            video_url = await client.upload(video_bytes, "video/mp4")')
                lines.append(f'            {field.name}_urls.append(video_url)')
                video_list_fields.append(field.name)
            elif field.python_type in ("ImageRef", "ImageRef | None"):
                lines.append(f"        {field.name}_base64 = (")
                lines.append(f"            await context.image_to_base64(self.{field.name})")
                lines.append(f"            if not self.{field.name}.is_empty()")
                lines.append("            else None")
                lines.append("        )")
                image_fields.append(field.name)
            elif field.python_type in ("VideoRef", "VideoRef | None"):
                lines.append(f"        {field.name}_url = (")
                lines.append(f"            await self._upload_asset_to_fal(client, self.{field.name}, context)")
                lines.append(f"            if not self.{field.name}.is_empty()")
                lines.append("            else None")
                lines.append("        )")
                video_fields.append(field.name)
        
        # Build arguments dict
        lines.append("        arguments = {")
        
        # Build a set of all extra fields that are part of nested structures
        # These will be handled with their main field
        nested_extra_fields = set()
        for main_field, (nested_key, extra_fields) in nested_asset_fields.items():
            for extra_field in extra_fields:
                nested_extra_fields.add(extra_field)
        
        for field in spec.input_fields:
            # Get the API parameter name (use original name if field was renamed)
            api_param_name = self._field_renames.get(field.name, field.name)
            
            # Skip extra fields that are part of nested structures - they'll be handled with the main field
            if field.name in nested_extra_fields:
                continue
            
            if field.name in image_fields:
                # Check if this image field has a nested structure
                if field.name in nested_asset_fields:
                    nested_key, extra_fields = nested_asset_fields[field.name]
                    # Build nested object
                    nested_lines = [f'            "{api_param_name}": {{']
                    nested_lines.append(f'                "{nested_key}": f"data:image/png;base64,{{{field.name}_base64}}" if {field.name}_base64 else None,')
                    for extra_field in extra_fields:
                        nested_lines.append(f'                "{extra_field}": self.{extra_field},')
                    nested_lines.append("            },")
                    lines.extend(nested_lines)
                else:
                    lines.append(
                        f'            "{api_param_name}": f"data:image/png;base64,{{{field.name}_base64}}" if {field.name}_base64 else None,'
                    )
            elif field.name in image_list_fields:
                lines.append(f'            "{api_param_name}": {field.name}_data_urls,')
            elif field.name in video_fields:
                # Check if this video field has a nested structure
                if field.name in nested_asset_fields:
                    nested_key, extra_fields = nested_asset_fields[field.name]
                    # Build nested object
                    nested_lines = [f'            "{api_param_name}": {{']
                    nested_lines.append(f'                "{nested_key}": {field.name}_url,')
                    for extra_field in extra_fields:
                        nested_lines.append(f'                "{extra_field}": self.{extra_field},')
                    nested_lines.append("            },")
                    lines.extend(nested_lines)
                else:
                    lines.append(f'            "{api_param_name}": {field.name}_url,')
            elif field.name in video_list_fields:
                lines.append(f'            "{api_param_name}": {field.name}_urls,')
            elif field.enum_ref:
                # Handle optional enums
                if not field.required and field.default_value == "None":
                    lines.append(f'            "{api_param_name}": self.{field.name}.value if self.{field.name} else None,')
                else:
                    lines.append(f'            "{api_param_name}": self.{field.name}.value,')
            elif self._is_base_type_list(field):
                # Serialize BaseType list items, excluding the internal 'type' field
                lines.append(f'            "{api_param_name}": [item.model_dump(exclude={{"type"}}) for item in self.{field.name}],')
            else:
                if field.python_type == "int | None":
                    lines.append(
                        f'            "{api_param_name}": (int(self.{field.name}.strip()) if isinstance(self.{field.name}, str) and self.{field.name}.strip() else self.{field.name}) if self.{field.name} is not None else None,'
                    )
                elif field.python_type == "float | None":
                    lines.append(
                        f'            "{api_param_name}": (float(self.{field.name}.strip()) if isinstance(self.{field.name}, str) and self.{field.name}.strip() else self.{field.name}) if self.{field.name} is not None else None,'
                    )
                else:
                    lines.append(f'            "{api_param_name}": self.{field.name},')
        
        lines.append("        }")
        lines.append("")
        
        # Filter out None values (recursively for nested dicts)
        lines.append("        # Remove None values")
        lines.append("        arguments = {k: v for k, v in arguments.items() if v is not None}")
        lines.append("        # Also filter nested dicts")
        lines.append("        for key in arguments:")
        lines.append("            if isinstance(arguments[key], dict):")
        lines.append("                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}")
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
            # Prioritize fields for better UX
            basic_fields = self._select_basic_fields(spec)
        
        fields_str = ", ".join(f'"{f}"' for f in basic_fields)
        
        return [
            "    @classmethod",
            "    def get_basic_fields(cls):",
            f"        return [{fields_str}]",
        ]
    
    def _select_basic_fields(self, spec: NodeSpec) -> list[str]:
        """Select the most important fields to show in basic mode.
        
        Prioritizes:
        1. Main input assets (image, video, audio)
        2. Text prompts
        3. Core generation parameters (resolution, aspect_ratio, duration)
        4. Other important fields
        
        Returns up to 5 field names.
        """
        candidates = []
        remaining = []
        
        for field in spec.input_fields:
            name_lower = field.name.lower()
            
            # Priority 1: Main input assets (not URLs, not lists)
            if field.python_type in ("ImageRef", "VideoRef", "AudioRef"):
                if name_lower in ("image", "video", "audio", "mask"):
                    candidates.append((0, field.name))
                else:
                    candidates.append((1, field.name))
            # Priority 2: Text prompts
            elif field.python_type == "str" and any(k in name_lower for k in ("prompt", "text")):
                candidates.append((2, field.name))
            # Priority 3: Core generation parameters
            elif field.python_type in ("Resolution", "AspectRatio") or any(
                field.enum_ref and field.enum_ref.endswith(k) for k in ("Resolution", "AspectRatio", "Duration")
            ):
                if not name_lower.endswith("_url"):  # Skip URL variants
                    candidates.append((3, field.name))
            # Priority 4: Other important scalar fields
            elif field.python_type in ("int", "float", "bool") and not name_lower.endswith(
                ("_seed", "_id", "_key", "_secret", "_steps", "_batch")
            ):
                candidates.append((4, field.name))
            else:
                remaining.append(field.name)
        
        # Sort by priority and then by original order for same priority
        candidates.sort(key=lambda x: (x[0], spec.input_fields.index(next(f for f in spec.input_fields if f.name == x[1]))))
        
        # Take up to 5 fields
        result = [name for _, name in candidates[:5]]
        
        # Fill with remaining fields if needed
        if len(result) < 5:
            result.extend(remaining[:5 - len(result)])
        
        return result
