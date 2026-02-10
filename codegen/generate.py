#!/usr/bin/env python3
"""
Main code generation script for FAL nodes.

This script generates FAL node code from OpenAPI schemas and config files.
"""

import asyncio
import argparse
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from codegen.schema_fetcher import SchemaFetcher
from codegen.schema_parser import SchemaParser
from codegen.node_generator import NodeGenerator


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


async def generate_node(
    endpoint_id: str,
    config_module: Optional[Any],
    fetcher: SchemaFetcher,
    parser: SchemaParser,
    generator: NodeGenerator,
    use_cache: bool = True
) -> tuple[str, str]:
    """
    Generate a single node.
    
    Args:
        endpoint_id: FAL endpoint ID
        config_module: Config module with overrides
        fetcher: Schema fetcher
        parser: Schema parser
        generator: Node generator
        use_cache: Whether to use cached schemas
        
    Returns:
        Tuple of (class_name, generated_code)
    """
    # Fetch schema
    print(f"Fetching schema for {endpoint_id}...")
    schema = await fetcher.fetch_schema(endpoint_id, use_cache=use_cache)
    
    # Parse schema
    print(f"Parsing schema for {endpoint_id}...")
    spec = parser.parse(schema)
    
    # Get config overrides
    config = {}
    if config_module and hasattr(config_module, "get_config"):
        config = config_module.get_config(endpoint_id)
    
    # Generate code
    print(f"Generating code for {endpoint_id}...")
    code = generator.generate(spec, config)
    
    return spec.class_name, code


async def generate_module(
    module_name: str,
    endpoints: list[str],
    output_dir: Path,
    use_cache: bool = True
):
    """
    Generate all nodes for a module.
    
    Args:
        module_name: Module name (e.g., 'image_to_video')
        endpoints: List of endpoint IDs
        output_dir: Output directory for generated code
        use_cache: Whether to use cached schemas
    """
    print(f"\n=== Generating module: {module_name} ===\n")
    
    # Initialize components
    fetcher = SchemaFetcher()
    parser = SchemaParser()
    generator = NodeGenerator()
    
    # Load config module
    config_path = Path(__file__).parent / "configs" / f"{module_name}.py"
    config_module = load_config_module(config_path)
    
    if config_module:
        print(f"Loaded config from {config_path}")
    else:
        print(f"No config found at {config_path}, using defaults")
    
    # Generate each node
    generated_nodes = []
    for endpoint_id in endpoints:
        try:
            class_name, code = await generate_node(
                endpoint_id,
                config_module,
                fetcher,
                parser,
                generator,
                use_cache
            )
            generated_nodes.append((class_name, code))
        except Exception as e:
            print(f"ERROR generating {endpoint_id}: {e}")
            continue
    
    # Write output file
    output_file = output_dir / f"{module_name}.py"
    print(f"\nWriting {len(generated_nodes)} nodes to {output_file}")
    
    # Determine which imports are actually needed by checking all generated code
    all_code = "\n".join(code for _, code in generated_nodes)
    # More precise detection - look for actual usage, not just string presence
    needs_any = (
        "-> Any" in all_code 
        or ": Any" in all_code 
        or "dict[str, Any]" in all_code
        or "list[Any]" in all_code
    )
    needs_image = "ImageRef" in all_code
    needs_video = "VideoRef" in all_code
    needs_audio = "AudioRef" in all_code
    needed_base_type_classes = sorted(
        bt_name
        for bt_name in NodeGenerator.KNOWN_BASE_TYPES
        if re.search(rf"\b{re.escape(bt_name)}\b", all_code)
    )
    
    # Extract and deduplicate enums from all generated code
    enums_seen = set()
    enums_to_write = []
    node_classes = []
    
    # Add shared enums from config if available
    if config_module and hasattr(config_module, "SHARED_ENUMS"):
        for enum_name, enum_def in config_module.SHARED_ENUMS.items():
            enums_seen.add(enum_name)
            lines = [f'class {enum_name}(str, Enum):']
            if "description" in enum_def:
                lines.append(f'    """{enum_def["description"]}"""')
            for value_name, value_str in enum_def["values"]:
                lines.append(f'    {value_name} = "{value_str}"')
            enums_to_write.append("\n".join(lines))
    
    for class_name, code in generated_nodes:
        lines = code.split("\n")
        # Skip import lines and extract enums
        enum_lines = []
        class_lines = []
        in_enum = False
        current_enum_name = None
        current_enum_lines = []
        skip_imports = True
        
        for line in lines:
            if skip_imports and (not line.strip() or line.startswith("from ") or line.startswith("import ")):
                continue
            skip_imports = False
            
            if line.startswith("class ") and "(Enum)" in line:
                # Start of an enum
                in_enum = True
                current_enum_name = line.split("(")[0].replace("class ", "").strip()
                current_enum_lines = [line]
            elif in_enum:
                current_enum_lines.append(line)
                # Check if we reached the end of the enum (empty line or next class)
                if not line.strip() or (line.startswith("class ") and "(Enum)" not in line and "(FALNode)" in line):
                    if current_enum_name and current_enum_name not in enums_seen:
                        enums_seen.add(current_enum_name)
                        enums_to_write.append("\n".join(current_enum_lines[:-1]))  # Exclude the line that broke the loop
                    in_enum = False
                    current_enum_name = None
                    current_enum_lines = []
                    # If this is the start of a class, we need to process it
                    if line.startswith("class ") and "(FALNode)" in line:
                        class_lines.append(line)
            else:
                class_lines.append(line)
        
        node_classes.append("\n".join(class_lines))
    
    with output_file.open("w") as f:
        # Write imports once at the top
        f.write("from enum import Enum\n")
        f.write("from pydantic import Field\n")
        if needs_any:
            f.write("from typing import Any\n")
        
        # Build asset types import
        asset_types = []
        if needs_image:
            asset_types.append("ImageRef")
        if needs_video:
            asset_types.append("VideoRef")
        if needs_audio:
            asset_types.append("AudioRef")
        
        if asset_types:
            f.write(f"from nodetool.metadata.types import {', '.join(asset_types)}\n")
        if needed_base_type_classes:
            f.write(f"from nodetool.nodes.fal.types import {', '.join(needed_base_type_classes)}\n")
        
        f.write("from nodetool.nodes.fal.fal_node import FALNode\n")
        f.write("from nodetool.workflows.processing_context import ProcessingContext\n")
        f.write("\n\n")
        
        # Write all unique enums
        for enum_code in enums_to_write:
            f.write(enum_code)
            f.write("\n\n\n")
        
        # Write all node classes
        for i, class_code in enumerate(node_classes):
            if i > 0:
                f.write("\n\n")
            f.write(class_code)
    
    print(f"✓ Generated {len(generated_nodes)} nodes for {module_name}")
    
    return generated_nodes


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate FAL nodes from OpenAPI schemas"
    )
    parser.add_argument(
        "--module",
        type=str,
        help="Module name to generate (e.g., 'image_to_video')"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        help="Single endpoint ID to generate"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated"),
        help="Output directory for generated code"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force fetch schemas without using cache"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate single endpoint
    if args.endpoint:
        fetcher = SchemaFetcher()
        schema_parser = SchemaParser()
        generator = NodeGenerator()
        
        class_name, code = await generate_node(
            args.endpoint,
            None,
            fetcher,
            schema_parser,
            generator,
            use_cache=not args.no_cache
        )
        
        output_file = args.output_dir / f"{class_name.lower()}.py"
        output_file.write_text(code)
        print(f"\n✓ Generated {class_name} to {output_file}")
        return
    
    # Generate module
    if args.module:
        # Load endpoints dynamically from config modules
        # This allows us to keep the generate script in sync with configs
        config_path = Path(__file__).parent / "configs" / f"{args.module}.py"
        config_module = load_config_module(config_path)
        
        if not config_module or not hasattr(config_module, "CONFIGS"):
            print(f"ERROR: No config found for module '{args.module}'")
            print(f"Available modules: Check codegen/configs/ directory")
            sys.exit(1)
        
        # Get all endpoint IDs from the config
        endpoints = list(config_module.CONFIGS.keys())
        
        if not endpoints:
            print(f"ERROR: No endpoints configured in module '{args.module}'")
            sys.exit(1)
        
        print(f"Loaded {len(endpoints)} endpoints from {args.module} config")
        
        await generate_module(
            args.module,
            endpoints,
            args.output_dir,
            use_cache=not args.no_cache
        )
        return
    
    print("ERROR: Must specify --module or --endpoint")
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
