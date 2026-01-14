# Repository Context

## Overview

`nodetool-fal` provides FAL integration nodes for the [NodeTool](https://github.com/nodetool-ai/nodetool) project. It depends on [nodetool-core](https://github.com/nodetool-ai/nodetool-core).

## Directory Structure

```
src/nodetool/
├── nodes/fal/                # All FAL node implementations
│   ├── fal_node.py           # Base class for all FAL nodes
│   ├── text_to_image.py      # Text-to-image models
│   ├── image_to_image.py     # Image transformation models
│   ├── image_to_video.py     # Image-to-video models
│   ├── text_to_video.py      # Text-to-video models
│   ├── text_to_audio.py      # Text-to-audio models
│   ├── speech_to_text.py     # Speech recognition models
│   └── llm.py                # LLM/text generation models
├── dsl/                      # Generated DSL code
└── package_metadata/         # Generated package metadata
```

## Key Files

### `fal_node.py`
Base class `FALNode` used by every node. Handles:
- Submitting requests to FAL endpoints
- Converting outputs to reference types
- Validating API key configuration

### Node files
Each node file contains multiple classes, each corresponding to a FAL endpoint. Nodes are grouped by modality (text-to-image, image-to-video, etc.).

## Commands

```bash
nodetool package scan
nodetool codegen

ruff check .
black --check .
```

## Code Style

- Python 3.11+ syntax
- Nodes inherit from `FALNode`
- Node attributes use `pydantic.Field`
- Each node has a descriptive docstring with tags and use cases
- Include `get_basic_fields` class method listing key fields
