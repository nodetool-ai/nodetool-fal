---
name: nodetool-fal-node-writer
description: Generate or update nodetool-fal node classes from fal.ai model docs. Use when the user provides one or more `https://fal.ai/models/.../llms.txt` URLs and wants production-style `FALNode` code for `src/nodetool/nodes/fal/*.py` plus optional enums, argument mapping, and output extraction.
---

# Nodetool FAL Node Writer

## Overview

Generate `nodetool-fal` node classes from Fal `llms.txt` model docs while matching current repository conventions.

## Workflow

1. Require at least one `llms.txt` URL.
2. Analyze current Fal implementations before generating code.
3. Parse each URL and generate a scaffold class.
4. Refine scaffold to match neighboring module style.
5. Validate with formatting/lint/tests when requested.

## Step 1: Require URL Input

Accept URLs in this format:

- `https://fal.ai/models/<publisher>/<model>/llms.txt`
- `https://fal.ai/models/<publisher>/<model>/<variant>/llms.txt`

If user gives a model page/API URL, convert it to `.../llms.txt` first.

## Step 2: Analyze Existing Fal Modules First

Always inspect all Fal node files before writing code so generated nodes follow current patterns, defaults, and naming.

Run:

```bash
rg -n "^class .*\(FALNode\)|async def process|get_basic_fields|application=|image_to_base64|client.upload|return .*Ref" src/nodetool/nodes/fal/*.py
```

Then read:

- `references/nodetool-fal-patterns.md` for mental model and conventions.
- `src/nodetool/nodes/fal/fal_node.py` for base request behavior.

## Step 3: Generate Scaffold from llms.txt

Use bundled script:

```bash
python3 scripts/scaffold_fal_node.py "<llms.txt-url>"
```

For multiple URLs:

```bash
python3 scripts/scaffold_fal_node.py "<url-1>" "<url-2>"
```

Script behavior:

- Fetch `llms.txt`.
- Extract model metadata (`Model ID`, category, input schema, output schema).
- Infer field types and enum candidates.
- Emit a `FALNode` class scaffold with `Field(...)`, `process(...)`, and `get_basic_fields`.

**Scaffold limitations** (fix manually or in post-processing):

- **List/array params** — The script infers a generic `list` type. It does not parse `list<SomeCompoundType>` (e.g. `list<KlingV3MultiPromptElement>`, `list<KlingV3ComboElementInput>`). Those appear in llms.txt as a single bullet (e.g. `elements`, `multi_prompt`) with type like `list<...>` and no per-item structure in the doc, so the script does not emit list-of-object handling.
- **Compound objects** — Params like `elements` (frontal_image_url + reference_image_urls, or video_url) are not broken down. The scaffold may omit them or treat them as a simple list; you must add proper fields and build the payload from the OpenAPI schema.
- **Conditional / paired params** — e.g. `shot_type` required when `multi_prompt` is provided. The script does not encode “only include when X is set”; add conditional logic when merging scaffold into the repo.
- **Single-option enums** — e.g. `shot_type` with only `"customize"`. The script may still emit an enum; if the API has only one valid value, you can omit the field and send the value when the paired feature (e.g. multi_prompt) is used.

## Step 4: Refine to Repository Style

After scaffold generation, **cross-check the OpenAPI** for the exact endpoint (`https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=<ENDPOINT_ID>`). Add or fix any **list-of-object** or **compound** params (e.g. `elements`, `multi_prompt`) and conditional fields (e.g. `shot_type` when `multi_prompt` is used) that the scaffold did not emit. Then enforce these conventions:

- Inherit from `FALNode`.
- Use media refs (`ImageRef`, `VideoRef`, `AudioRef`, `Model3DRef`) for media inputs/outputs.
- Convert image refs with `context.image_to_base64(...)` and send as data URIs.
- Upload video/audio bytes when endpoint requires uploaded files rather than URLs.
- Build `arguments` with optional-field guards.
- Extract outputs using existing patterns (`res["video"]["url"]`, `res["images"][0]["url"]`, etc.).
- Implement `get_basic_fields()` with top-priority fields.

If an endpoint shape is ambiguous, match the nearest existing node in the same modality file (`text_to_image.py`, `image_to_video.py`, `text_to_video.py`, etc.).

## Step 5: Placement and Follow-up

Add the class to the correct node file under:

- `src/nodetool/nodes/fal/`

Then suggest the standard follow-up commands:

```bash
nodetool package scan
nodetool codegen
ruff check .
black --check .
pytest -q
```

Use `nodetool package scan` and `nodetool codegen` whenever node definitions changed.

---

## Important: Validation checklist (from AGENTS.md)

Before considering a node done, verify:

1. **Field names match the API exactly** — Check OpenAPI: `https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=<ENDPOINT_ID>`
2. **Only send fields the API supports** — Variants (e.g. V3 vs O3) have different parameters.
3. **Never send `None` to the API** — Optional strings from the UI can be `None`; sending `null` causes `input_value_error`. Guard with:
   ```python
   if self.negative_prompt is not None and self.negative_prompt.strip():
       arguments["negative_prompt"] = self.negative_prompt
   ```
4. **Compound objects** — Include every required sub-field the schema expects.
5. **Enums** — Do not reuse across API variants without checking; each endpoint may support different values.
6. **Variants** — Treat Standard/Pro or V3/O3 as separate APIs; re-check schema for the exact endpoint.
7. **Docstring Format**: First line describes the model, second line contains comma-separated tags, followed by use cases
8. **Field Descriptions**: Use clear, descriptive text from the OpenAPI schema
9. **Default Values**: Set sensible defaults based on schema defaults
10. **Optional Fields**: Handle conditional arguments properly
11. **Error Handling**: Always assert expected output fields exist
