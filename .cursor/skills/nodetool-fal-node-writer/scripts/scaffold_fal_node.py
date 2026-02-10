#!/usr/bin/env python3
"""Generate nodetool-fal FALNode scaffolds from fal.ai llms.txt URLs."""

from __future__ import annotations

import argparse
import re
import textwrap
from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.request import urlopen


@dataclass
class Param:
    original_name: str
    field_name: str
    schema_type: str
    required: bool
    description: str
    default_raw: str | None
    options: list[str]
    # True when type is list<SomeCompoundType> / array of X; scaffold cannot build payload
    compound_list: bool = False


@dataclass
class ModelSpec:
    title: str
    model_id: str
    category: str
    params: list[Param]
    output_name: str
    output_type: str


MEDIA_HINTS = {
    "frame": "ImageRef",
    "image": "ImageRef",
    "video": "VideoRef",
    "audio": "AudioRef",
    "3d": "Model3DRef",
    "model": "Model3DRef",
    "mesh": "Model3DRef",
}


def pascal_case(value: str) -> str:
    parts = re.split(r"[^a-zA-Z0-9]+", value)
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


def enum_member(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.upper()).strip("_")
    if not cleaned:
        cleaned = "VALUE"
    if cleaned[0].isdigit():
        cleaned = f"V_{cleaned}"
    return cleaned


def clean_literal(value: str) -> str:
    value = value.strip().strip("`").strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        value = value[1:-1]
    return value.strip()


def is_llms_url(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme in {"http", "https"}
        and parsed.netloc == "fal.ai"
        and parsed.path.endswith("/llms.txt")
    )


def fetch_text(url: str) -> str:
    with urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_model_spec(markdown: str) -> ModelSpec:
    title_match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Fal Model"

    model_id_match = re.search(r"\*\*Model ID\*\*:\s*`([^`]+)`", markdown)
    if not model_id_match:
        raise ValueError("Could not find Model ID in llms.txt")
    model_id = model_id_match.group(1).strip()

    category_match = re.search(r"\*\*Category\*\*:\s*([^\n]+)", markdown)
    category = category_match.group(1).strip() if category_match else "unknown"

    input_section = re.search(
        r"### Input Schema\n(.*?)\n### Output Schema", markdown, re.DOTALL
    )
    if not input_section:
        raise ValueError("Could not find Input Schema section")
    params = parse_params(input_section.group(1))

    output_section = re.search(
        r"### Output Schema\n(.*?)(\n## Usage Examples|\n## Additional Resources|\Z)",
        markdown,
        re.DOTALL,
    )
    output_name, output_type = parse_output(output_section.group(1) if output_section else "")

    return ModelSpec(
        title=title,
        model_id=model_id,
        category=category,
        params=params,
        output_name=output_name,
        output_type=output_type,
    )


def parse_params(section: str) -> list[Param]:
    params: list[Param] = []
    pattern = re.compile(
        r"\n- \*\*`(?P<name>[^`]+)`\*\* \(`(?P<type>[^`]+)`\s*,\s*_(?P<required>required|optional)_\):\n"
        r"(?P<body>.*?)(?=\n- \*\*`|\Z)",
        re.DOTALL,
    )
    for m in pattern.finditer("\n" + section):
        name = m.group("name").strip()
        schema_type = m.group("type").strip()
        required = m.group("required") == "required"
        body = m.group("body").strip()

        default_match = re.search(r"- Default:\s*`?\"?([^`\"\n]+)\"?`?", body)
        default_raw = clean_literal(default_match.group(1)) if default_match else None

        options_match = re.search(r"- Options:\s*(.+)", body)
        options = []
        if options_match:
            options = [clean_literal(opt) for opt in options_match.group(1).split(",")]

        description_line = ""
        for line in body.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("-"):
                description_line = stripped.rstrip(".")
                break
        if not description_line:
            description_line = f"Input for {name}"

        field_name = map_field_name(name, schema_type, description_line)
        # list<ObjectType> (not list<string>) = compound list; we emit field + TODO, no auto-assign
        _is_list = re.match(r"list\s*<", schema_type) or re.search(
            r"array\s+of\s+\w+", schema_type, re.IGNORECASE
        )
        _simple_inner = any(
            s in schema_type.lower() for s in ("string", "integer", "number", "boolean")
        )
        compound_list = bool(_is_list) and not _simple_inner

        params.append(
            Param(
                original_name=name,
                field_name=field_name,
                schema_type=schema_type,
                required=required,
                description=description_line,
                default_raw=default_raw,
                options=options,
                compound_list=compound_list,
            )
        )
    return params


def parse_output(section: str) -> tuple[str, str]:
    m = re.search(r"- \*\*`([^`]+)`\*\* \(`([^`]+)`\s*,\s*_required_\)", section)
    if not m:
        return "result", "dict[str, Any]"
    return m.group(1).strip(), m.group(2).strip()


def map_field_name(name: str, schema_type: str, desc: str) -> str:
    if name.endswith("_url") and schema_type == "string":
        return name[: -len("_url")]
    return name


def detect_media_ref(name: str, desc: str) -> str | None:
    probe = f"{name} {desc}".lower()
    for hint, ref in MEDIA_HINTS.items():
        if hint in probe:
            return ref
    return None


def infer_field_type(param: Param) -> tuple[str, bool]:
    media_ref = detect_media_ref(param.original_name, param.description)
    if media_ref and param.original_name.endswith("_url"):
        base = media_ref
    elif param.options:
        base = pascal_case(param.field_name) + "Enum"
    elif "integer" in param.schema_type:
        base = "int"
    elif "float" in param.schema_type or "number" in param.schema_type:
        base = "float"
    elif "boolean" in param.schema_type:
        base = "bool"
    elif "list" in param.schema_type or "array" in param.schema_type:
        base = "list[dict[str, Any]]" if getattr(param, "compound_list", False) else "list"
    else:
        base = "str"

    optional = (not param.required) and (param.default_raw is None)
    if optional and base in {"ImageRef", "VideoRef", "AudioRef", "Model3DRef"}:
        return f"{base} | None", True
    if optional and base not in {"str", "int", "float", "bool", "list"}:
        return f"{base} | None", True
    if optional:
        return f"{base} | None", True
    return base, False


def field_default(param: Param, base_type: str, optional: bool) -> str:
    if optional:
        return "None"

    if base_type == "ImageRef":
        return "ImageRef()"
    if base_type == "VideoRef":
        return "VideoRef()"
    if base_type == "AudioRef":
        return "AudioRef()"
    if base_type == "Model3DRef":
        return "Model3DRef()"

    if param.options:
        enum_name = pascal_case(param.field_name) + "Enum"
        if param.default_raw:
            for opt in param.options:
                if opt == param.default_raw:
                    return f"{enum_name}.{enum_member(opt)}"
        return f"{enum_name}.{enum_member(param.options[0])}"

    if param.default_raw is not None:
        if base_type == "int":
            return str(int(float(param.default_raw)))
        if base_type == "float":
            return str(float(param.default_raw))
        if base_type == "bool":
            return "True" if param.default_raw.lower() == "true" else "False"
        return repr(param.default_raw)

    if base_type == "str":
        return '""'
    if base_type == "bool":
        return "False"
    if base_type == "int":
        return "0"
    if base_type == "float":
        return "0.0"
    if base_type == "list" or base_type.startswith("list["):
        return "[]"
    return "None"


def render_enum(param: Param) -> str:
    enum_name = pascal_case(param.field_name) + "Enum"
    lines = [f"class {enum_name}(Enum):"]
    for opt in param.options:
        lines.append(f"    {enum_member(opt)} = {opt!r}")
    return "\n".join(lines)


def pick_return_type(output_name: str, output_type: str) -> tuple[str, str]:
    probe = f"{output_name} {output_type}".lower()
    if "video" in probe:
        return "VideoRef", 'return VideoRef(uri=res["video"]["url"])'
    if "audio" in probe:
        return "AudioRef", 'return AudioRef(uri=res["audio"]["url"])'
    if "image" in probe and "list" in probe:
        return "ImageRef", 'return ImageRef(uri=res["images"][0]["url"])'
    if "image" in probe:
        return "ImageRef", 'return ImageRef(uri=res["image"]["url"])'
    if "mesh" in probe or "model" in probe or "3d" in probe:
        return "Model3DRef", 'return Model3DRef(uri=res.get("model", {}).get("url", ""))'
    return "dict[str, Any]", "return res"


def render_class(spec: ModelSpec, source_url: str) -> str:
    class_name = pascal_case(spec.model_id.split("/")[-1]) or "FalGeneratedNode"

    enums = [render_enum(p) for p in spec.params if p.options]

    field_rows: list[str] = []
    pre_rows: list[str] = []
    dict_rows: list[str] = []
    post_rows: list[str] = []
    basic_fields: list[str] = []

    for p in spec.params:
        full_type, optional = infer_field_type(p)
        base_type = full_type.replace(" | None", "")
        default = field_default(p, base_type, optional)
        desc = p.description or f"Input for {p.original_name}"
        # Override type for known compound lists so node has usable inputs
        if getattr(p, "compound_list", False) and p.original_name == "elements":
            full_type = "list[ImageRef]"
            default = "[]"
        elif getattr(p, "compound_list", False) and p.original_name == "multi_prompt":
            full_type = "list[dict[str, Any]]"
            default = "[]"

        field_rows.append(
            f"    {p.field_name}: {full_type} = Field(default={default}, description={desc!r})"
        )

        media_ref = detect_media_ref(p.original_name, p.description)
        if p.required and len(basic_fields) < 3:
            basic_fields.append(p.field_name)

        if p.original_name.endswith("_url") and media_ref == "ImageRef":
            if optional:
                post_rows.extend(
                    [
                        f"        if self.{p.field_name} is not None:",
                        f"            {p.field_name}_base64 = await context.image_to_base64(self.{p.field_name})",
                        f"            arguments[{p.original_name!r}] = f\"data:image/png;base64,{{{p.field_name}_base64}}\"",
                    ]
                )
            else:
                pre_rows.append(
                    f"        {p.field_name}_base64 = await context.image_to_base64(self.{p.field_name})"
                )
                dict_rows.append(
                    f'            "{p.original_name}": f"data:image/png;base64,{{{p.field_name}_base64}}",'
                )
            continue

        # Compound list: generate working code for known shapes, else TODO
        if getattr(p, "compound_list", False):
            if p.original_name == "multi_prompt":
                post_rows.extend(
                    [
                        "        if self.multi_prompt:",
                        '            arguments["multi_prompt"] = [{"prompt": str(d.get("prompt", "")), "duration": str(d.get("duration", "5"))} for d in self.multi_prompt]',
                        '            arguments["shot_type"] = "customize"',
                    ]
                )
                continue
            if p.original_name == "elements":
                post_rows.extend(
                    [
                        "        if self.elements:",
                        "            elements_urls = []",
                        "            for ref in self.elements:",
                        "                if ref.uri:",
                        "                    b = await context.image_to_base64(ref)",
                        "                    elements_urls.append(f\"data:image/png;base64,{b}\")",
                        "            if elements_urls:",
                        "                arguments[\"elements\"] = [{\"frontal_image_url\": u, \"reference_image_urls\": [u]} for u in elements_urls]",
                    ]
                )
                continue
            post_rows.append(
                f"        # TODO: build {p.original_name!r} from node fields; see OpenAPI schema for shape"
            )
            continue

        value_expr = (
            f"self.{p.field_name}.value"
            if p.options and not base_type.endswith("None")
            else f"self.{p.field_name}"
        )
        if p.options and optional:
            value_expr = f"self.{p.field_name}.value"

        if optional:
            # For list params, only send when non-empty to avoid sending []
            guard = (
                f"self.{p.field_name}"
                if base_type == "list" or base_type.startswith("list[")
                else f"self.{p.field_name} is not None"
            )
            post_rows.extend(
                [
                    f"        if {guard}:",
                    f"            arguments[{p.original_name!r}] = {value_expr}",
                ]
            )
        else:
            dict_rows.append(f'            "{p.original_name}": {value_expr},')

    if not basic_fields:
        basic_fields = [p.field_name for p in spec.params[:3]]

    return_type, output_line = pick_return_type(spec.output_name, spec.output_type)

    imports = [
        "from enum import Enum",
        "from typing import Any",
        "from pydantic import Field",
        "from nodetool.metadata.types import ImageRef, VideoRef, AudioRef, Model3DRef",
        "from nodetool.nodes.fal.fal_node import FALNode",
        "from nodetool.workflows.processing_context import ProcessingContext",
    ]

    args_block = "\n".join(dict_rows)
    arguments_builder = (
        "        arguments: dict[str, Any] = {\n"
        + args_block
        + "\n        }"
        if dict_rows
        else "        arguments: dict[str, Any] = {}"
    )

    code = f"""# Source llms.txt: {source_url}
# Model category: {spec.category}

{textwrap.dedent(chr(10).join(imports))}

{chr(10).join(enums)}

class {class_name}(FALNode):
    \"\"\"
    Generated scaffold for {spec.title}.
    Edit descriptions, defaults, and tags before shipping.
    \"\"\"

{chr(10).join(field_rows)}

    async def process(self, context: ProcessingContext) -> {return_type}:
{chr(10).join(pre_rows)}
{arguments_builder}
{chr(10).join(post_rows)}

        res = await self.submit_request(
            context=context,
            application={spec.model_id!r},
            arguments=arguments,
        )
        {output_line}

    @classmethod
    def get_basic_fields(cls):
        return {basic_fields!r}
"""
    return normalize_blank_lines(code)


def normalize_blank_lines(code: str) -> str:
    code = re.sub(r"\n{3,}", "\n\n", code)
    return code.strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urls", nargs="+", help="fal.ai llms.txt URL(s)")
    args = parser.parse_args()

    for i, url in enumerate(args.urls):
        if not is_llms_url(url):
            raise SystemExit(f"Invalid llms.txt URL: {url}")

        md = fetch_text(url)
        spec = parse_model_spec(md)
        output = render_class(spec, url)

        if i:
            print("\n" + "#" * 80 + "\n")
        print(output)


if __name__ == "__main__":
    main()
