# Node Creation Guide

This guide explains how to add new FAL models to the nodetool-fal repository.

## Step 1: Discover New Models

Use the FAL model catalog (https://fal.ai/models) or OpenAPI schemas from FAL endpoints to find models that are:
- Official/featured providers
- Recently released or updated
- Popular in the community
- Not yet implemented in `src/nodetool/nodes/fal/`

## Step 2: Choose the Right Category

Pick the correct file based on the endpoint modality:
- `text_to_image.py` - Text-to-image models
- `image_to_image.py` - Image transformation models
- `image_to_video.py` - Image-to-video models
- `text_to_video.py` - Text-to-video models
- `text_to_audio.py` - Text-to-audio models
- `speech_to_text.py` - Speech recognition models
- `llm.py` - Text/LLM models

## Step 3: Implement the Node Class

Each node must:
- Inherit from `FALNode`
- Use `pydantic.Field` for attributes
- Include a docstring with a short description, tags, and 5 use cases
- Implement `async def process(self, context: ProcessingContext)`
- Provide `get_basic_fields` with 3-5 key fields

## Step 4: Handle Inputs

Common input patterns:

**Images**
```python
image_base64 = await context.image_to_base64(self.image)
arguments["image_url"] = f"data:image/png;base64,{image_base64}"
```

**Videos**
```python
client = self.get_client(context)
video_bytes = await context.asset_to_bytes(self.video)
arguments["video_url"] = await client.upload(video_bytes, "video/mp4")
```

**Audio**
```python
client = self.get_client(context)
audio_bytes = await context.asset_to_bytes(self.audio)
arguments["audio_url"] = await client.upload(audio_bytes, "audio/mp3")
```

## Step 5: Handle Outputs

Match the output schema:
- Image outputs → `ImageRef`
- Video outputs → `VideoRef`
- Audio outputs → `AudioRef`
- Text outputs → `str`
- Complex outputs → custom `return_type` dictionary

Always assert expected keys exist in the response.

## Step 6: Regenerate Metadata and DSL

```bash
nodetool package scan
nodetool codegen
```

## Step 7: Verify Formatting

```bash
ruff check .
black --check .
```

## Checklist

- [ ] Model is active on FAL
- [ ] Endpoint ID is correct
- [ ] Node class is in the right file
- [ ] Docstring follows the required format
- [ ] `get_basic_fields` includes key inputs
- [ ] Metadata and DSL regenerated
- [ ] Formatting checks pass
