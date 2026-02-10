#!/usr/bin/env python3
"""
Script to merge generated config additions into existing config files.
"""

import re
from pathlib import Path


def merge_configs(config_file: Path, additions_file: Path) -> str:
    """Merge generated configs into an existing config file."""
    
    # Read existing config
    with open(config_file, 'r') as f:
        existing = f.read()
    
    # Read additions
    with open(additions_file, 'r') as f:
        additions_content = f.read()
    
    # Extract the additions (skip header comments)
    additions_lines = additions_content.split('\n')
    additions_text = '\n'.join(line for line in additions_lines if not line.strip().startswith('#'))
    
    # Find the closing brace of CONFIGS dict
    # Look for the pattern: },\n}
    match = re.search(r'(\s+},)\n(\})\n\n\ndef get_config', existing)
    
    if not match:
        print(f"  Warning: Could not find insertion point in {config_file.name}")
        return existing
    
    # Insert before the closing brace
    insertion_point = match.start(2)
    
    # Merge
    new_content = (
        existing[:insertion_point] +
        additions_text.rstrip() + '\n' +
        existing[insertion_point:]
    )
    
    return new_content


def main():
    """Main entry point."""
    
    additions_dir = Path('/tmp/generated_configs')
    configs_dir = Path('codegen/configs')
    
    # Map of addition files to config files
    mappings = {
        'image_to_image_additions.txt': 'image_to_image.py',
        'text_to_image_additions.txt': 'text_to_image.py',
        'image_to_video_additions.txt': 'image_to_video.py',
        'text_to_video_additions.txt': 'text_to_video.py',
        'video_to_video_additions.txt': 'video_to_video.py',
        'vision_additions.txt': 'vision.py',
        'text_to_audio_additions.txt': 'text_to_audio.py',
    }
    
    # Create new config files for categories that don't exist yet
    new_categories = {
        'training_additions.txt': 'training.py',
        'image_to_3d_additions.txt': 'image_to_3d.py',
        'audio_to_video_additions.txt': 'audio_to_video.py',
        '3d_to_3d_additions.txt': '3d_to_3d.py',
    }
    
    print("=== Merging Generated Configs ===\n")
    
    # Merge into existing files
    for additions_file, config_file in mappings.items():
        additions_path = additions_dir / additions_file
        config_path = configs_dir / config_file
        
        if not additions_path.exists():
            print(f"Skipping {additions_file} - not found")
            continue
        
        if not config_path.exists():
            print(f"Skipping {config_file} - does not exist")
            continue
        
        print(f"Processing {config_file}...")
        
        try:
            new_content = merge_configs(config_path, additions_path)
            
            # Write back
            with open(config_path, 'w') as f:
                f.write(new_content)
            
            print(f"  ✅ Merged additions into {config_file}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Create new config files
    print("\n=== Creating New Config Files ===\n")
    
    template = '''"""
Configuration for {module_name} module.

This config file defines overrides and customizations for {module_name} nodes.
"""

from typing import Any


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {{
{configs}
}}


def get_config(endpoint_id: str) -> dict[str, Any]:
    """
    Get configuration for an endpoint.
    
    Args:
        endpoint_id: FAL endpoint ID
        
    Returns:
        Configuration dictionary
    """
    return CONFIGS.get(endpoint_id, {{}})
'''
    
    for additions_file, config_file in new_categories.items():
        additions_path = additions_dir / additions_file
        config_path = configs_dir / config_file
        
        if not additions_path.exists():
            print(f"Skipping {additions_file} - not found")
            continue
        
        if config_path.exists():
            print(f"Skipping {config_file} - already exists")
            continue
        
        print(f"Creating {config_file}...")
        
        try:
            # Read additions
            with open(additions_path, 'r') as f:
                additions_content = f.read()
            
            # Extract configs (skip header)
            additions_lines = additions_content.split('\n')
            config_text = '\n'.join(line for line in additions_lines if not line.strip().startswith('#'))
            
            # Create module name
            module_name = config_file.replace('.py', '').replace('_', '-')
            
            # Generate file content
            new_content = template.format(
                module_name=module_name,
                configs=config_text.strip()
            )
            
            # Write file
            with open(config_path, 'w') as f:
                f.write(new_content)
            
            print(f"  ✅ Created {config_file}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n=== Summary ===")
    print("Configuration files updated!")
    print("\nNext steps:")
    print("1. Review the updated config files")
    print("2. Run: python codegen/generate.py --module <name> --output-dir src/nodetool/nodes/fal")
    print("3. Run: nodetool package scan")
    print("4. Run: nodetool codegen")


if __name__ == '__main__':
    main()
