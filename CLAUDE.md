# Contributor Guidelines

This repository provides additional nodes for the [nodetool](https://github.com/nodetool-ai/nodetool) project and depends on [nodetool-core](https://github.com/nodetool-ai/nodetool-core).

## Code Style

- Use **Python 3.11+** syntax.
- All nodes live under `src/nodetool/nodes/fal` and must inherit from `FALNode`.
- Node attributes are defined with `pydantic.Field` and async `process` methods should return the appropriate reference type.
- Each node must contain a short docstring describing the model and several example use cases.
- Provide a `get_basic_fields` class method listing the most relevant fields

## Adding a New FAL Node from OpenAPI Schema

When implementing a new FAL node from an OpenAPI specification, follow this systematic approach:

### 1. Extract Key Information from OpenAPI Schema

From the OpenAPI JSON, identify:
- **Endpoint ID**: `x-fal-metadata.endpointId` (e.g., `"fal-ai/luma-dream-machine/image-to-video"`)
- **Input Schema**: Main input schema name (e.g., `LumaDreamMachineImageToVideoInput`)
- **Output Schema**: Main output schema name (e.g., `LumaDreamMachineImageToVideoOutput`)
- **Required/Optional Fields**: Check `required` array in schema properties

### 2. Node Implementation Template

```python
from pydantic import Field
from enum import Enum
from nodetool.metadata.types import ImageRef, VideoRef, AudioRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext

# Create enums for constrained values
class AspectRatio(Enum):
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_4_3 = "4:3"
    RATIO_3_4 = "3:4"
    RATIO_21_9 = "21:9"
    RATIO_9_21 = "9:21"

class YourModelName(FALNode):
    """
    Brief description of what the model does and its capabilities.
    comma, separated, tags, describing, functionality, category
    
    Use cases:
    - First use case example
    - Second use case example
    - Third use case example
    - Fourth use case example
    - Fifth use case example
    """
    
    # Map OpenAPI properties to pydantic Fields
    prompt: str = Field(
        default="", description="Description from OpenAPI schema"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Input image description"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9,
        description="The aspect ratio of the generated output",
    )
    loop: bool = Field(
        default=False, description="Boolean field description"
    )
    
    async def process(self, context: ProcessingContext) -> VideoRef:
        # Handle different input types
        if hasattr(self, 'image') and self.image.uri:
            image_base64 = await context.image_to_base64(self.image)
            
        # Build arguments matching OpenAPI schema
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "loop": self.loop,
        }
        
        # Handle image inputs
        if hasattr(self, 'image') and self.image.uri:
            arguments["image_url"] = f"data:image/png;base64,{image_base64}"
            
        # Handle optional fields
        if hasattr(self, 'optional_field') and self.optional_field:
            arguments["optional_field"] = self.optional_field
            
        # Submit request
        res = await self.submit_request(
            context=context,
            application="endpoint-id-from-openapi-schema",
            arguments=arguments,
        )
        
        # Handle output based on schema
        assert "expected_output_key" in res
        return VideoRef(uri=res["expected_output_key"]["url"])
    
    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image", "aspect_ratio"]  # Most important fields
```

### 3. Type Mapping Guide

| OpenAPI Type | Python Type | nodetool Type |
|--------------|-------------|---------------|
| `string` | `str` | `str` |
| `integer` | `int` | `int` |
| `number` | `float` | `float` |
| `boolean` | `bool` | `bool` |
| `array` | `list[T]` | `list[T]` |
| `enum` | `Enum` | Custom `Enum` class |
| Image URL | `str` | `ImageRef` |
| Video URL | `str` | `VideoRef` |
| Audio URL | `str` | `AudioRef` |

### 4. Input Processing Patterns

**For Images:**
```python
image_base64 = await context.image_to_base64(self.image)
arguments["image_url"] = f"data:image/png;base64,{image_base64}"
```

**For Videos:**
```python
client = self.get_client(context)
video_bytes = await context.asset_to_bytes(self.video)
video_url = await client.upload(video_bytes, "video/mp4")
arguments["video_url"] = video_url
```

**For Audio:**
```python
client = self.get_client(context)  
audio_bytes = await context.asset_to_bytes(self.audio)
audio_url = await client.upload(audio_bytes, "audio/mp3")
arguments["audio_url"] = audio_url
```

### 5. Output Processing Patterns

**Single Image Output:**
```python
assert res["images"] is not None
assert len(res["images"]) > 0
return ImageRef(uri=res["images"][0]["url"])
```

**Single Video Output:**
```python
assert "video" in res
return VideoRef(uri=res["video"]["url"])
```

**Complex/Multiple Outputs:**
```python
@classmethod
def return_type(cls):
    return {
        "text": str,
        "chunks": list[dict],
        "metadata": dict,
    }

# In process method:
return {
    "text": result["text"],
    "chunks": result["chunks"], 
    "metadata": result.get("metadata", {}),
}
```

### 6. Special Considerations

**Conditional Parameters:** Handle optional fields that should only be included when set:
```python
if self.seed != -1:
    arguments["seed"] = self.seed
```

**File Organization:** Place nodes in appropriate files:
- `text_to_image.py` - Text-to-image models
- `image_to_image.py` - Image transformation models  
- `image_to_video.py` - Image-to-video models
- `speech_to_text.py` - Speech recognition models
- Create new files for new categories as needed

**Error Handling:** Always validate expected outputs:
```python
assert "expected_key" in res
assert res["expected_key"] is not None
```

### 7. Documentation Requirements

1. **Docstring Format**: 
   - Line 1: Clear description of functionality
   - Line 2: Comma-separated tags for searchability  
   - Lines 3+: 5 concrete use cases

2. **Field Descriptions**: Use exact or adapted descriptions from OpenAPI schema

3. **get_basic_fields**: Return 3-5 most important fields for UI display

## Commands

After adding or changing nodes run these commands to generate metadata and DSL.

```bash
nodetool package scan
nodetool codegen
```

## Linting and Tests

Before submitting a pull request, run the following checks:

```bash
ruff check .
black --check .
pytest -q
```

Formatting issues or lint errors should be fixed before committing. Test coverage is expected to be added when applicable.
