# Contributor Guidelines

This repository provides additional nodes for the [nodetool](https://github.com/nodetool-ai/nodetool) project and depends on [nodetool-core](https://github.com/nodetool-ai/nodetool-core).

## Code Style

- Use **Python 3.11** syntax.
- All nodes live under `src/nodetool/nodes/fal` and must inherit from `FALNode`.
- Node attributes are defined with `pydantic.Field` and async `process` methods should return the appropriate reference type.
- Each node must contain a short docstring describing the model and several example use cases.
- Provide a `get_basic_fields` class method listing the most relevant fields

## Adding a New FAL Node

### Step 1: Analyze the OpenAPI Schema

When adding a new FAL node from an OpenAPI schema, first extract the key information:

1. **Endpoint ID**: Found in `x-fal-metadata.endpointId` (e.g., `fal-ai/luma-dream-machine/image-to-video`)
2. **Input Schema**: Look for the main input schema (e.g., `LumaDreamMachineImageToVideoInput`)
3. **Output Schema**: Look for the main output schema (e.g., `LumaDreamMachineImageToVideoOutput`)

### Step 2: Create the Node Class

Create a new class in the appropriate file under `src/nodetool/nodes/fal/`:

```python
from pydantic import Field
from enum import Enum
from nodetool.metadata.types import ImageRef, VideoRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext

class AspectRatio(Enum):
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    # ... add other ratios from schema

class LumaDreamMachine(FALNode):
    """
    Generate video clips from your images using Luma Dream Machine v1.5. Supports various aspect ratios and optional end-frame blending.
    video, generation, animation, blending, aspect-ratio, img2vid, image-to-video

    Use cases:
    - Create seamless video loops
    - Generate video transitions
    - Transform images into animations
    - Create motion graphics
    - Produce video content
    """

    # Map OpenAPI schema properties to pydantic Fields
    image: ImageRef = Field(
        default=ImageRef(), description="The image to transform into a video"
    )
    prompt: str = Field(
        default="", description="A description of the desired video motion and style"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9,
        description="The aspect ratio of the generated video",
    )
    loop: bool = Field(
        default=False,
        description="Whether the video should loop (end blends with beginning)",
    )
    end_image: ImageRef | None = Field(
        default=None, description="Optional image to blend the end of the video with"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        # Convert images to base64
        image_base64 = await context.image_to_base64(self.image)

        # Build arguments dict matching OpenAPI schema
        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "loop": self.loop,
        }

        # Handle optional fields
        if self.end_image:
            end_image_base64 = await context.image_to_base64(self.end_image)
            arguments["end_image_url"] = f"data:image/png;base64,{end_image_base64}"

        # Submit request using the endpoint ID
        res = await self.submit_request(
            context=context,
            application="fal-ai/luma-dream-machine/image-to-video",
            arguments=arguments,
        )
        
        # Extract result based on output schema
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt", "aspect_ratio"]
```

### Step 3: Map OpenAPI Types to Python Types

Common mappings:

- `string` → `str`
- `integer` → `int`
- `number` → `float`
- `boolean` → `bool`
- `array` → `list[T]`
- `enum` → Create Python `Enum` class
- File references → `ImageRef`, `VideoRef`, `AudioRef`

### Step 4: Handle Input Processing

For different input types:

- **Images**: Use `await context.image_to_base64(self.image)` and format as `data:image/png;base64,{base64}`
- **Videos**: Use `await context.asset_to_bytes(self.video)` and upload with `await client.upload(video_bytes, "video/mp4")`
- **Audio**: Use `await context.asset_to_bytes(self.audio)` and upload with `await client.upload(audio_bytes, "audio/mp3")`

### Step 5: Handle Output Processing

Based on the output schema:

- **Single image**: `return ImageRef(uri=res["images"][0]["url"])`
- **Single video**: `return VideoRef(uri=res["video"]["url"])`
- **Multiple outputs**: Return appropriate reference type or dict
- **Complex outputs**: Define custom return type with `@classmethod def return_type(cls)`

### Step 6: Add Enums for Constrained Values

For schema properties with `enum` values:

```python
class AspectRatio(Enum):
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_4_3 = "4:3"
    RATIO_3_4 = "3:4"
```

### Best Practices

1. **Docstring Format**: First line describes the model, second line contains comma-separated tags, followed by use cases
2. **Field Descriptions**: Use clear, descriptive text from the OpenAPI schema
3. **Default Values**: Set sensible defaults based on schema defaults
4. **Optional Fields**: Handle conditional arguments properly
5. **Error Handling**: Always assert expected output fields exist
6. **Naming**: Use descriptive class names that reflect the model/functionality

## ⚠️ Python Environment (IMPORTANT)

**Local Development:** Use the conda `nodetool` environment. Do not use system Python.

```bash
conda activate nodetool
# then run commands normally
```

**GitHub CI / Copilot Agent:** Uses standard Python 3.11 with pip. Dependencies are pre-installed via `.github/workflows/copilot-setup-steps.yml`. Run commands directly without conda.

## Commands

After adding or changing nodes run these commands to generate metadata and DSL.

```bash
nodetool package scan
nodetool codegen
```

### DSL Generation Process

When you add a new FAL node, `nodetool codegen` automatically generates corresponding DSL classes in `src/nodetool/dsl/fal/`. Here's how the process works:

1. **Package Scan**: `nodetool package scan` discovers your new node classes and generates metadata
2. **Code Generation**: `nodetool codegen` creates DSL wrapper classes based on the node metadata

**Example**: If you create a new node `LumaDreamMachine` in `src/nodetool/nodes/fal/image_to_video.py`, the codegen will automatically create a corresponding DSL class in `src/nodetool/dsl/fal/image_to_video.py`:

```python
class LumaDreamMachine(GraphNode):
    """
    Generate video clips from your images using Luma Dream Machine v1.5. Supports various aspect ratios and optional end-frame blending.
    video, generation, animation, blending, aspect-ratio, img2vid, image-to-video

    Use cases:
    - Create seamless video loops
    - Generate video transitions
    - Transform images into animations
    - Create motion graphics
    - Produce video content
    """

    image: ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=ImageRef(), description="The image to transform into a video"
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="A description of the desired video motion and style"
    )
    aspect_ratio: str | GraphNode | tuple[GraphNode, str] = Field(
        default="16:9", description="The aspect ratio of the generated video"
    )
    loop: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=False, description="Whether the video should loop (end blends with beginning)"
    )

    @classmethod
    def get_node_type(cls):
        return "fal.image_to_video.LumaDreamMachine"
```

**Key DSL Features**:

- All field types become unions with `GraphNode` and `tuple[GraphNode, str]` for graph connectivity
- Enum fields are converted to string types in DSL
- The `get_node_type()` method returns the fully qualified node path
- Docstrings and field descriptions are preserved

**File Organization**: DSL files mirror the node file structure:

- `src/nodetool/nodes/fal/image_to_video.py` → `src/nodetool/dsl/fal/image_to_video.py`
- `src/nodetool/nodes/fal/text_to_image.py` → `src/nodetool/dsl/fal/text_to_image.py`

## Linting and Tests

Before submitting a pull request, run the following checks:

```bash
ruff check .
black --check .
pytest -q
```

Formatting issues or lint errors should be fixed before committing. Test coverage is expected to be added when applicable.
