# FAL Node Code Generation Framework

This directory contains the code generation framework for automatically generating FAL nodes from OpenAPI specifications.

## Overview

The framework fetches OpenAPI schemas from FAL.ai, parses them to extract input/output definitions, and generates Python node classes with proper typing, documentation, and behavior.

## Components

### 1. `schema_fetcher.py`
Fetches and caches OpenAPI schemas from FAL API endpoints.

- Caches schemas in `.codegen_cache/` for faster regeneration
- Uses `httpx` for async HTTP requests
- Handles endpoint ID to URL conversion

### 2. `schema_parser.py`
Parses OpenAPI schemas into structured node specifications.

- Extracts input/output schemas from OpenAPI paths
- Resolves `$ref` references and `allOf` schemas
- Generates enum definitions from constrained string values
- Maps JSON Schema types to Python types
- Detects ImageRef/VideoRef/AudioRef fields automatically

### 3. `node_generator.py`
Generates Python code from node specifications.

- Creates class definitions with proper inheritance
- Generates field definitions with pydantic Field
- Creates async process methods with correct API calls
- Handles enum renaming and value remapping
- Supports field renaming while preserving API parameter names
- Applies configuration overrides

### 4. `generate.py`
Main CLI script for code generation.

```bash
# Generate a single node
python codegen/generate.py --endpoint "fal-ai/flux/dev" --output-dir generated

# Generate all nodes for a module
python codegen/generate.py --module image_to_video --output-dir generated

# Force refresh schemas (bypass cache)
python codegen/generate.py --module image_to_video --no-cache --output-dir generated
```

### 5. `configs/`
Module-specific configuration files that override and customize code generation.

Each config file defines a `CONFIGS` dict mapping endpoint IDs to override settings.

## Configuration System

Configs are Python files that allow fine-grained control over generated code:

```python
CONFIGS = {
    "fal-ai/endpoint-id": {
        # Basic overrides
        "class_name": "CustomClassName",
        "docstring": "Custom description",
        "tags": ["tag1", "tag2"],
        "use_cases": ["Use case 1", "Use case 2", ...],
        
        # Field customization
        "field_overrides": {
            "field_name": {
                "name": "renamed_field",  # Rename field (API still uses original)
                "description": "Custom description",
                "python_type": "CustomType",
                "default_value": "CustomDefault()"
            }
        },
        
        # Enum customization
        "enum_overrides": {
            "Resolution": "CustomResolution"  # Rename enum class
        },
        "enum_value_overrides": {
            "Resolution": {
                "720P": "RES_720P"  # Rename enum value
            }
        },
        
        # Display fields
        "basic_fields": ["field1", "field2", "field3"]
    }
}
```

## Type Mapping

The parser automatically maps JSON Schema types to Python/nodetool types:

| JSON Schema | Python Type | Notes |
|-------------|-------------|-------|
| `string` | `str` | |
| `string` with `enum` | Custom `Enum` | Generated enum class |
| `string` (image URL) | `ImageRef` | Auto-detected from field name/description |
| `string` (video URL) | `VideoRef` | Auto-detected from field name/description |
| `string` (audio URL) | `AudioRef` | Auto-detected from field name/description |
| `integer` | `int` | |
| `number` | `float` | |
| `boolean` | `bool` | |
| `array` | `list[T]` | |
| `object` | `dict` | |

## Output Type Detection

The framework automatically detects the correct output type:

- Looks for `video` property → `VideoRef`
- Looks for `images` property → `ImageRef`
- Looks for `audio` property → `AudioRef`
- Multiple properties → `dict[str, Any]`

## Workflow

1. **Fetch Schema**: Get OpenAPI spec from FAL.ai for endpoint
2. **Parse Schema**: Extract input/output definitions and enums
3. **Load Config**: Apply module-specific overrides
4. **Generate Code**: Create Python class with all components
5. **Write File**: Output to module file with shared imports

## Adding New Modules

1. Create config file: `configs/{module_name}.py`
2. Copy from `configs/template.py` and customize
3. Add endpoint IDs to `generate.py` module_endpoints dict
4. Run: `python codegen/generate.py --module {module_name} --output-dir generated`

## Best Practices

### Field Naming
- Use descriptive names that match existing patterns
- Rename `image_url` → `image` for consistency
- Keep API parameter name in rename mapping

### Enum Naming
- Use PascalCase for enum classes: `PixverseV56Resolution`
- Use SCREAMING_SNAKE_CASE for values: `RES_720P`, `FIVE_SECONDS`
- Prefix numeric values: `VALUE_5` → `FIVE_SECONDS`

### Documentation
- First line: Clear description of functionality
- Second line: Comma-separated tags for searchability
- Use cases: 5 concrete examples

### Configuration
- Start minimal, add overrides only as needed
- Test generated code before adding more config
- Keep configs maintainable - prefer automated detection over manual overrides

## Troubleshooting

### 404 errors
- Check endpoint ID is correct
- Verify endpoint exists in all_models.json
- Try fetching schema directly in browser

### Wrong output type
- Check if output schema detection is finding correct schema
- Verify schema has expected properties (video, images, etc.)
- Add logging to parser to debug

### Enum issues
- Check enum value format in OpenAPI schema
- Verify value override mappings are correct
- Test enum generation independently

## Future Enhancements

- [ ] Support for custom process method implementations
- [ ] Automatic field ordering based on importance
- [ ] Validation of generated code before writing
- [ ] Comparison tool for generated vs existing nodes
- [ ] Batch generation for all endpoints
- [ ] Auto-detection of config needs from existing code
