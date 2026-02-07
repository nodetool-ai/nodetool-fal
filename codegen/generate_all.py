#!/usr/bin/env python3
"""
Generate nodes for all modules with configurations.
"""

import asyncio
import subprocess
import sys
from pathlib import Path


MODULES = [
    'image_to_image',
    'text_to_image',
    'image_to_video',
    'text_to_video',
    'video_to_video',
    'vision',
    'text_to_audio',
    'text_to_speech',
    'audio_to_audio',
    'speech_to_text',
    'llm',
    'training',
    'image_to_3d',
    'audio_to_video',
    '3d_to_3d',
]


async def generate_module(module_name: str, output_dir: Path) -> bool:
    """Generate nodes for a single module."""
    print(f"\n{'=' * 80}")
    print(f"Generating module: {module_name}")
    print('=' * 80)
    
    cmd = [
        sys.executable,
        'codegen/generate.py',
        '--module', module_name,
        '--output-dir', str(output_dir)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout per module
        )
        
        if result.returncode == 0:
            print(f"✅ Successfully generated {module_name}")
            return True
        else:
            print(f"❌ Error generating {module_name}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏱️  Timeout generating {module_name}")
        return False
    except Exception as e:
        print(f"❌ Exception generating {module_name}: {e}")
        return False


async def main():
    """Main entry point."""
    output_dir = Path('src/nodetool/nodes/fal')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("BULK NODE GENERATION")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Modules to generate: {len(MODULES)}")
    print()
    
    # Generate all modules sequentially (to avoid overwhelming the API)
    results = {}
    for module in MODULES:
        success = await generate_module(module, output_dir)
        results[module] = success
    
    # Summary
    print("\n" + "=" * 80)
    print("GENERATION SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for success in results.values() if success)
    failed = len(results) - successful
    
    print(f"Total modules: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed modules:")
        for module, success in results.items():
            if not success:
                print(f"  - {module}")
    
    print("\n" + "=" * 80)
    
    return failed == 0


if __name__ == '__main__':
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
