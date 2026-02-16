#!/usr/bin/env python3
"""
Generate provider model lists from codegen/configs and OpenAPI schemas.

This script reads the codegen config modules (the source of truth for
which FAL endpoints are supported) and generates a Python module containing
model lists for each model type (ImageModel, VideoModel, etc.) that the
FalProvider can use.
"""

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, Optional

# Config module name -> (model_type_class, list_name, supported_tasks)
MODULE_MAP: dict[str, tuple[str, str, list[str]]] = {
    "text_to_image": ("ImageModel", "IMAGE_MODELS", ["text-to-image"]),
    "image_to_image": ("ImageModel", "IMAGE_MODELS", ["image-to-image"]),
    "text_to_video": ("VideoModel", "VIDEO_MODELS", ["text-to-video"]),
    "image_to_video": ("VideoModel", "VIDEO_MODELS", ["image-to-video"]),
    "video_to_video": ("VideoModel", "VIDEO_MODELS", ["video-to-video"]),
    "audio_to_video": ("VideoModel", "VIDEO_MODELS", ["audio-to-video"]),
    "text_to_speech": ("TTSModel", "TTS_MODELS", []),
    "text_to_audio": ("TTSModel", "TTS_MODELS", []),
    "speech_to_text": ("ASRModel", "ASR_MODELS", []),
    "audio_to_text": ("ASRModel", "ASR_MODELS", []),
    "image_to_3d": ("Model3DModel", "MODEL_3D_MODELS", ["image-to-3d"]),
    "text_to_3d": ("Model3DModel", "MODEL_3D_MODELS", ["text-to-3d"]),
    "3d_to_3d": ("Model3DModel", "MODEL_3D_MODELS", ["3d-to-3d"]),
}


def load_config_module(config_path: Path) -> Optional[Any]:
    """Load a config module from a Python file."""
    if not config_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _class_name_to_title(class_name: str) -> str:
    """Convert PascalCase class name to a human-readable title.

    Example: 'FluxV1ProUltra' -> 'Flux V1 Pro Ultra'
    """
    # Insert space before each uppercase letter that follows a lowercase letter
    # or before a sequence of uppercase letters followed by a lowercase letter
    title = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", class_name)
    title = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", title)
    return title


def _extract_model_name(endpoint_id: str, config: dict[str, Any]) -> str:
    """Extract a human-readable model name from a config entry.

    Priority:
    1. First sentence of the docstring (up to the first period)
    2. Derived from class_name
    3. The endpoint_id itself
    """
    if "class_name" in config:
        return _class_name_to_title(config["class_name"])
    return endpoint_id


def collect_models_from_configs(
    configs_dir: Path,
) -> dict[str, list[tuple[str, str, str, list[str]]]]:
    """Read all codegen config modules and collect model entries.

    Returns:
        dict mapping list_name -> list of (endpoint_id, name, model_class, tasks)
    """
    result: dict[str, list[tuple[str, str, str, list[str]]]] = {}
    seen: dict[str, set[str]] = {}

    for config_file in sorted(configs_dir.glob("*.py")):
        module_name = config_file.stem
        if module_name in ("template", "__init__"):
            continue
        if module_name not in MODULE_MAP:
            continue

        module = load_config_module(config_file)
        if module is None or not hasattr(module, "CONFIGS"):
            continue

        model_class, list_name, tasks = MODULE_MAP[module_name]
        configs: dict[str, dict[str, Any]] = module.CONFIGS

        if list_name not in result:
            result[list_name] = []
            seen[list_name] = set()

        for endpoint_id, cfg in configs.items():
            if endpoint_id in seen[list_name]:
                continue
            seen[list_name].add(endpoint_id)
            name = _extract_model_name(endpoint_id, cfg)
            result[list_name].append((endpoint_id, name, model_class, tasks))

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
    configs_dir = Path(__file__).parent / "configs"
    output_file = repo_root / "src" / "nodetool" / "fal" / "generated_models.py"

    if not configs_dir.exists():
        print(f"ERROR: {configs_dir} not found")
        sys.exit(1)

    print(f"Reading codegen configs from {configs_dir}...")
    model_lists = collect_models_from_configs(configs_dir)

    total = 0
    for list_name, entries in model_lists.items():
        print(f"  {list_name}: {len(entries)} models")
        total += len(entries)
    print(f"  Total: {total} models")

    print(f"Writing generated code to {output_file}...")
    code = generate_code(model_lists)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(code)

    print("✓ Done")


if __name__ == "__main__":
    main()
