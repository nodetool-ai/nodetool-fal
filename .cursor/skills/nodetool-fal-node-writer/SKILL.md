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
rg -n "^class .*\(FALNode\)|async def process|get_basic_fields|application=|image_to_base64|client.upload|return .*Ref" /Users/mg/workspace/nodetool-fal/src/nodetool/nodes/fal/*.py
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

## Step 4: Refine to Repository Style

After scaffold generation, enforce these conventions:

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
