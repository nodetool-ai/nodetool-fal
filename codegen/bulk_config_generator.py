#!/usr/bin/env python3
"""
Bulk configuration generator for FAL endpoints.

This script helps generate configuration templates for multiple endpoints at once,
making it easier to reach coverage goals.
"""

import json
import asyncio
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from codegen.schema_fetcher import SchemaFetcher
from codegen.schema_parser import SchemaParser


async def generate_config_template(endpoint_id: str, model_info: dict) -> dict:
    """Generate a basic config template for an endpoint."""
    
    # Create a sensible class name from endpoint ID
    parts = endpoint_id.replace('fal-ai/', '').split('/')
    
    # Try to create a meaningful class name
    if len(parts) >= 2:
        # e.g., "flux-2/flash" -> "Flux2Flash"
        # e.g., "ltx-2/image-to-video" -> "LTX2ImageToVideo"
        class_name = ''.join(word.title().replace('-', '') for word in parts)
    else:
        class_name = ''.join(word.title().replace('-', '') for word in parts)
    
    # Extract category for appropriate tags
    category = model_info.get('category', '')
    tags = []
    
    if 'text-to-image' in category:
        tags = ['generation', 'text-to-image', 'ai-art']
    elif 'image-to-image' in category:
        tags = ['editing', 'transformation', 'image-to-image']
    elif 'text-to-video' in category:
        tags = ['video', 'generation', 'text-to-video']
    elif 'image-to-video' in category:
        tags = ['video', 'animation', 'image-to-video']
    elif 'video-to-video' in category:
        tags = ['video', 'editing', 'video-to-video']
    
    # Try to fetch schema to get better information
    try:
        fetcher = SchemaFetcher()
        schema = await fetcher.fetch_schema(endpoint_id, use_cache=True)
        parser = SchemaParser()
        spec = parser.parse(schema)
        
        # Use parsed info if available
        if spec.input_fields:
            # Extract some field names for tags
            field_names = [f.name for f in spec.input_fields[:3]]
            for name in field_names:
                if 'prompt' in name.lower():
                    if 'prompt' not in tags:
                        tags.append('prompt-based')
                elif 'image' in name.lower():
                    if 'image-based' not in tags:
                        tags.append('image-based')
                elif 'video' in name.lower():
                    if 'video-based' not in tags:
                        tags.append('video-based')
    except Exception as e:
        print(f"  Warning: Could not fetch schema for {endpoint_id}: {e}")
    
    return {
        'class_name': class_name,
        'docstring': f"{model_info['title']}. {model_info.get('shortDescription', '')}",
        'tags': tags,
        'use_cases': [
            'Professional content creation',
            'Creative projects',
            'Automated workflows',
            'Batch processing',
            'Rapid prototyping'
        ]
    }


async def main():
    """Main entry point."""
    # Load all models
    with open('all_models.json', 'r') as f:
        all_models = json.load(f)
    
    # Load existing configs
    configured = set()
    for config_file in Path('codegen/configs').glob('*.py'):
        if config_file.name in ['__init__.py', 'template.py']:
            continue
        
        import importlib.util
        spec = importlib.util.spec_from_file_location('config', config_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'CONFIGS'):
            configured.update(module.CONFIGS.keys())
    
    print(f"Total models: {len(all_models)}")
    print(f"Already configured: {len(configured)}")
    print(f"Remaining: {len(all_models) - len(configured)}")
    
    # Group unconfigured by category
    unconfigured = defaultdict(list)
    for model in all_models:
        if model['id'] not in configured:
            unconfigured[model['category']].append(model)
    
    # Priority categories for 50% coverage
    priorities = {
        'image-to-image': 106,  # Need 106 more
        'video-to-video': 51,    # Need 51 more
        'text-to-image': 39,     # Need 39 more
        'image-to-video': 35,    # Need 35 more
        'text-to-video': 34,     # Need 34 more
    }
    
    print("\n=== Generating config templates for priority categories ===\n")
    
    for category, target_count in priorities.items():
        models = unconfigured[category]
        print(f"\n{category}: Need {target_count} more (have {len(models)} unconfigured)")
        
        # Take the first N models
        selected = models[:target_count]
        
        print(f"Generating templates for {len(selected)} models...")
        
        configs = {}
        for i, model in enumerate(selected):
            endpoint_id = model['id']
            print(f"  {i+1}/{len(selected)}: {endpoint_id}")
            
            try:
                config = await generate_config_template(endpoint_id, model)
                configs[endpoint_id] = config
            except Exception as e:
                print(f"    Error: {e}")
                # Fallback to basic config
                configs[endpoint_id] = {
                    'class_name': 'Generated' + str(i),
                    'docstring': model['title'],
                    'tags': [category.replace('-', '_')],
                    'use_cases': ['Generated use case']
                }
        
        # Save to a temp file for review
        output_file = Path(f'codegen/configs_generated/{category}.json')
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(configs, f, indent=2)
        
        print(f"  Saved to {output_file}")
    
    print("\n=== Summary ===")
    print(f"Total templates generated: {sum(priorities.values())}")
    print(f"Files saved to: codegen/configs_generated/")
    print("\nNext steps:")
    print("1. Review generated templates in codegen/configs_generated/")
    print("2. Convert JSON templates to Python CONFIGS format")
    print("3. Add to appropriate config files in codegen/configs/")
    print("4. Run code generation: python codegen/generate.py --module <name>")


if __name__ == '__main__':
    asyncio.run(main())
