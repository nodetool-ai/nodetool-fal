# Nodetool Fal Patterns

## Scope Reviewed

All modules under:

- `/Users/mg/workspace/nodetool-fal/src/nodetool/nodes/fal/*.py`
- `/Users/mg/workspace/nodetool-fal/src/nodetool/dsl/fal/*.py`
- `/Users/mg/workspace/nodetool-fal/src/nodetool/nodes/fal/fal_node.py`

## Core Mental Model

- Runtime nodes live in `src/nodetool/nodes/fal/*.py` and inherit `FALNode`.
- `FALNode.submit_request(...)` wraps `fal_client.AsyncClient.submit(...)` and consumes queue events.
- Nodes expose pydantic fields and implement `async def process(self, context) -> <type>`.
- Nodes pass an `arguments` dict to `self.submit_request(...)` using Fal model IDs such as `fal-ai/flux-pro/kontext`.
- Most nodes add `get_basic_fields()` for editor UI defaults.

## Type Patterns

- Image output: `ImageRef(uri=res["images"][0]["url"])` or `ImageRef(uri=res["image"]["url"])`
- Video output: `VideoRef(uri=res["video"]["url"])`
- Audio output: `AudioRef(uri=res["audio"]["url"])` or endpoint-specific key
- 3D output: `Model3DRef(uri=res.get("model", {}).get("url", ""))` style
- Multi-output endpoints use `TypedDict` + dict return.

## Input Conversion Patterns

- ImageRef input:
  - `image_base64 = await context.image_to_base64(self.image)`
  - `arguments["image_url"] = f"data:image/png;base64,{image_base64}"`
- VideoRef / AudioRef input:
  - Use `await context.asset_to_bytes(...)`
  - Upload with `client = await self.get_client(context)` then `await client.upload(...)`
  - Set argument to uploaded URL.

## Enum and Defaults Patterns

- Constrained options become Python `Enum` classes.
- Enum fields use `.value` in `arguments`.
- Optional numeric seeds often use sentinel `-1` and conditionally include in args.
- Optional prompts commonly use empty string default and conditional inclusion.

## File Placement Heuristics

- `text_to_image.py`: prompt to image models.
- `image_to_image.py`: image editing/restoration.
- `image_to_video.py`: image-conditioned video generation.
- `text_to_video.py`: prompt-conditioned video generation.
- `text_to_audio.py`: TTS/music/audio generation.
- `speech_to_text.py`: ASR/transcription.
- `vision.py`: captioning/OCR/understanding.
- `segmentation.py`: masks/pose/depth style outputs.
- `model3d.py`: 2D-to-3D generation.
- `video_processing.py`: upscaling/interpolation/lipsync/video transforms.

## DSL Relationship

- DSL classes under `src/nodetool/dsl/fal/*.py` are generated wrappers.
- Do not manually handcraft DSL unless intentionally bypassing `nodetool codegen`.
- Standard flow after adding a node:
  - `nodetool package scan`
  - `nodetool codegen`

## Quality Checklist

- Include concise docstring with tags + use cases.
- Include field descriptions derived from schema text.
- Guard optional arguments.
- Assert/handle expected output keys.
- Return the correct ref type.
- Add `get_basic_fields()` with most useful 1-4 fields.
