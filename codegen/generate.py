#!/usr/bin/env python3
"""
Main code generation script for FAL nodes.

This script generates FAL node code from OpenAPI schemas and config files.
"""

import asyncio
import argparse
import importlib.util
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
    output_file = output_dir / f"{module_name}_generated.py"
    print(f"\nWriting {len(generated_nodes)} nodes to {output_file}")
    
    with output_file.open("w") as f:
        # Write imports once at the top
        f.write("from enum import Enum\n")
        f.write("from pydantic import Field\n")
        f.write("from typing import Any\n")
        f.write("from nodetool.metadata.types import ImageRef, VideoRef, AudioRef\n")
        f.write("from nodetool.nodes.fal.fal_node import FALNode\n")
        f.write("from nodetool.workflows.processing_context import ProcessingContext\n")
        f.write("\n\n")
        
        for i, (class_name, code) in enumerate(generated_nodes):
            if i > 0:
                f.write("\n\n")
            # Remove imports from individual node code
            lines = code.split("\n")
            # Skip import lines
            start_idx = 0
            for idx, line in enumerate(lines):
                if not line.strip() or line.startswith("from ") or line.startswith("import "):
                    start_idx = idx + 1
                else:
                    break
            f.write("\n".join(lines[start_idx:]))
    
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
        # Define endpoints for each module
        module_endpoints = {
            "image_to_video": [
                "fal-ai/pixverse/v5.6/image-to-video",
                "fal-ai/luma-dream-machine/image-to-video",
            ],
            # Add more modules here
        }
        
        if args.module not in module_endpoints:
            print(f"ERROR: Unknown module '{args.module}'")
            print(f"Available modules: {', '.join(module_endpoints.keys())}")
            sys.exit(1)
        
        await generate_module(
            args.module,
            module_endpoints[args.module],
            args.output_dir,
            use_cache=not args.no_cache
        )
        return
    
    print("ERROR: Must specify --module or --endpoint")
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
