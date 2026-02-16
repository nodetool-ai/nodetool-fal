#!/usr/bin/env python3
"""
Generate provider model lists from all_models.json.

This script reads the all_models.json file and generates a Python module
containing model lists for each model type (ImageModel, VideoModel, etc.)
that the FalProvider can use.
"""

import json
import sys
from pathlib import Path

# Category -> (model_type_class, list_name, supported_tasks)
CATEGORY_MAP: dict[str, tuple[str, str, list[str]]] = {
    "text-to-image": ("ImageModel", "IMAGE_MODELS", ["text-to-image"]),
    "image-to-image": ("ImageModel", "IMAGE_MODELS", ["image-to-image"]),
    "text-to-video": ("VideoModel", "VIDEO_MODELS", ["text-to-video"]),
    "image-to-video": ("VideoModel", "VIDEO_MODELS", ["image-to-video"]),
    "video-to-video": ("VideoModel", "VIDEO_MODELS", ["video-to-video"]),
    "text-to-speech": ("TTSModel", "TTS_MODELS", []),
    "text-to-audio": ("TTSModel", "TTS_MODELS", []),
    "speech-to-text": ("ASRModel", "ASR_MODELS", []),
    "audio-to-text": ("ASRModel", "ASR_MODELS", []),
    "image-to-3d": ("Model3DModel", "MODEL_3D_MODELS", ["image-to-3d"]),
    "text-to-3d": ("Model3DModel", "MODEL_3D_MODELS", ["text-to-3d"]),
    "3d-to-3d": ("Model3DModel", "MODEL_3D_MODELS", ["3d-to-3d"]),
}


def load_models(models_file: Path) -> list[dict]:
    """Load models from all_models.json."""
    with models_file.open() as f:
        return json.load(f)


def generate_model_lists(
    models: list[dict],
) -> dict[str, list[tuple[str, str, str, list[str]]]]:
    """
    Group models by list name.

    Returns:
        dict mapping list_name -> list of (id, title, model_class, supported_tasks)
    """
    result: dict[str, list[tuple[str, str, str, list[str]]]] = {}
    seen: dict[str, set[str]] = {}

    for model in models:
        category = model.get("category", "")
        if category not in CATEGORY_MAP:
            continue

        model_class, list_name, tasks = CATEGORY_MAP[category]
        model_id = model["id"]
        title = model.get("title", model_id)

        if list_name not in result:
            result[list_name] = []
            seen[list_name] = set()

        if model_id not in seen[list_name]:
            seen[list_name].add(model_id)
            result[list_name].append((model_id, title, model_class, tasks))

    return result


def generate_code(model_lists: dict[str, list[tuple[str, str, str, list[str]]]]) -> str:
    """Generate Python code for the model lists module."""
    lines: list[str] = []

    lines.append('"""')
    lines.append("Auto-generated model lists for the FAL provider.")
    lines.append("")
    lines.append(
        "DO NOT EDIT — regenerate with:  python codegen/generate_provider_models.py"
    )
    lines.append('"""')
    lines.append("")

    # Collect which model classes we need
    needed_classes: set[str] = set()
    for entries in model_lists.values():
        for _, _, model_class, _ in entries:
            needed_classes.add(model_class)

    # Build import line
    sorted_classes = sorted(needed_classes)
    lines.append(
        f"from nodetool.metadata.types import Provider, {', '.join(sorted_classes)}"
    )
    lines.append("")
    lines.append("")

    # Determine the order of lists to emit
    list_order = [
        "IMAGE_MODELS",
        "VIDEO_MODELS",
        "TTS_MODELS",
        "ASR_MODELS",
        "MODEL_3D_MODELS",
    ]

    for list_name in list_order:
        if list_name not in model_lists:
            continue

        entries = model_lists[list_name]
        model_class = entries[0][2]

        lines.append(f"{list_name}: list[{model_class}] = [")
        for model_id, title, mc, tasks in entries:
            # Escape any quotes in title
            safe_title = title.replace('"', '\\"')
            if tasks:
                tasks_str = repr(tasks)
                lines.append(
                    f'    {mc}(id="{model_id}", name="{safe_title}", provider=Provider.FalAI, supported_tasks={tasks_str}),'
                )
            else:
                lines.append(
                    f'    {mc}(id="{model_id}", name="{safe_title}", provider=Provider.FalAI),'
                )
        lines.append("]")
        lines.append("")
        lines.append("")

    return "\n".join(lines)


def main():
    """Main entry point."""
    repo_root = Path(__file__).parent.parent
    models_file = repo_root / "all_models.json"
    output_file = repo_root / "src" / "nodetool" / "fal" / "generated_models.py"

    if not models_file.exists():
        print(f"ERROR: {models_file} not found")
        sys.exit(1)

    print(f"Loading models from {models_file}...")
    models = load_models(models_file)
    print(f"Loaded {len(models)} models")

    print("Generating model lists...")
    model_lists = generate_model_lists(models)

    for list_name, entries in model_lists.items():
        print(f"  {list_name}: {len(entries)} models")

    print(f"Writing generated code to {output_file}...")
    code = generate_code(model_lists)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(code)

    print("✓ Done")


if __name__ == "__main__":
    main()
